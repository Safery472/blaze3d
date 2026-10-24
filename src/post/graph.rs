// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 后处理图.
//!
//! [`PostProcessGraph`] 是 [`super::chain::PostProcessChain`] 的一般化.
//! 比线性链多出两种结构: 分支 (一个 pass 的输出被多个 pass 读取) 与
//! 合流 (一个 pass 读取多个输入).
//!
//! # 与 `PostProcessChain` 的关系
//!
//! [`super::chain::PostProcessChain`] 是**线性**的: 每个 pass 的输入只能
//! 是外部纹理或**上一个** pass. 这足以覆盖绝大多数后处理需求 (暗角 →
//! 色调映射 → 抗锯齿), 但表达不了:
//!
//! - **分支**: 同一个中间结果喂给两个不同的效果 (例如「原图」同时
//!   送去做泛光和做景深).
//! - **合流**: 一个 pass 需要两个输入 (例如泛光需要「原图 + 模糊图」,
//!   合成 pass 需要「场景 + 泛光」).
//!
//! 本模块用**有向无环图 (DAG)** 表达这类结构: 节点是 pass, 边是数据
//! 依赖.
//!
//! # 节点与依赖
//!
//! [`PostProcessGraph::add_node`] 添加一个节点并返回 [`NodeId`]. 节点的
//! 输入用 [`NodeInput`] 描述:
//!
//! - [`NodeInput::External`] —— `draw_into_frame` 时传入的外部 bind group.
//! - `NodeInput::Node(id)` —— 另一个节点的输出.
//!
//! 一个节点可以有**多个**输入 —— 但每个输入对应一个独立的 bind group
//! 槽位 (group 2, 3, ...). 这与 [`super::chain::PostProcessChain`] 的
//! 「group 1 是输入」约定不同: 图上第一个输入固定绑定到 group 1 (兼容
//! 线性链的约定), 后续输入依次绑定到 group 2+.
//!
//! # 执行顺序
//!
//! [`PostProcessGraph::draw_into_frame`] 内部做**拓扑排序** —— 保证每个
//! 节点都在它的所有依赖之后执行. 排序结果会被缓存: 图结构不变时不会
//! 重复计算.
//!
//! # 中间目标
//!
//! 每个「被其他节点读取」的节点输出都需要一个中间目标. 目标池按尺寸
//! 与格式自动重建. 与线性链的 ping-pong 策略不同, 图无法预知目标数
//! —— 池按需扩展, 数量等于「需要被读取的节点数」.
//!
//! # 输出节点
//!
//! [`PostProcessGraph::set_output`] 指定哪个节点写入最终目标. 输出节点
//! **不允许被其他节点引用** —— 否则它的结果去向有二义性 (既写
//! `final_target` 又要作为中间输入). 违反时
//! [`PostProcessGraph::draw_into_frame`] 返回
//! [`crate::pipeline::Error::InvalidNodeRef`].
//!
//! 显式设置输出节点时, id 必须指向一个已添加的节点 —— 否则
//! `draw_into_frame` 同样返回 [`crate::pipeline::Error::InvalidNodeRef`].
//! 未显式设置时默认取拓扑序最后一个节点, 该 id 必然有效.
//!
//! # 环检测
//!
//! 添加节点时**不**校验环 —— 校验推迟到 `draw_into_frame` (拓扑排序时)
//! 执行. 这样 `add_node` 不会因为「先建节点后连边」的合法用法而失败.
//! 检测到环时返回 [`crate::pipeline::Error::CycleDetected`].

use std::collections::HashMap;
use std::sync::Arc;

use crate::pipeline::{RenderTarget, Result, Target};
use crate::shader::Shader;
use crate::state::{ColorFormat, LayoutHandle, RenderType};
use crate::system::Frame;

use super::chain::PostProcessInput;
use super::quad::FullscreenQuad;

// ================ 节点标识 ================

/// 图节点标识.
///
/// 由 [`PostProcessGraph::add_node`] 分配, 从 0 开始逐个递增. 图内的
/// 其他节点通过 [`NodeInput`] 引用它.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// 一个节点的输入来源.
///
/// 主输入由 [`PostProcessGraph::add_node`] 的 `input` 参数给出, 额外的
/// 输入见 [`ExtraInput`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeInput {
	/// 外部输入 —— `draw_into_frame` 时传入的 bind group.
	External,
	/// 引用另一个节点的输出.
	Node(NodeId),
}

/// 一个节点的**额外**输入.
///
/// 节点的第一个输入 (binding group 1) 由 [`PostProcessGraph::add_node`]
/// 的 `input` 参数给出; 需要更多输入时用
/// [`PostProcessGraph::add_input`].
///
/// 每个额外输入占用一个独立的 bind group 槽位, 从 group 2 开始.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExtraInput {
	/// 该额外输入的数据来源.
	pub source: NodeInput,
	/// 绑定到哪个 group.
	///
	/// 必须 >= 2 —— group 0 是 globals, group 1 是第一个输入.
	pub group: u32,
}

// ================ 图节点 ================

/// 图中的一个 pass 节点.
///
/// 绘制时主输入绑定到 group 1, 额外输入按各自的 `group` 字段绑定到
/// group 2 及之后.
pub struct PostProcessNode {
	/// 该节点的管线状态.
	pub render_type: Arc<RenderType>,
	/// 该节点使用的后处理着色器.
	pub shader: Shader,
	/// 管线布局.
	///
	/// **必须**遵循:
	///
	/// - group 0 —— globals.
	/// - group 1 —— 主输入 (来自 `input`).
	/// - group 2+ —— 额外输入 (来自 `extra_inputs`), 按 `group` 字段.
	pub layout: LayoutHandle,
	/// 主输入来源.
	pub input: NodeInput,
	/// 额外输入, 绑定到 group 2 及之后.
	pub extra_inputs: Vec<ExtraInput>,
}

// ================ 图 ================

/// 后处理图.
///
/// # 使用流程
///
/// 1. 启动阶段: [`Self::new`] 创建图, 然后反复调用 [`Self::add_node`]
///    添加节点并保存返回的 [`NodeId`].
/// 2. 节点需要多个输入时调用 [`Self::add_input`] 补充.
/// 3. 标记输出节点: [`Self::set_output`] —— 不标记时最后一个添加的
///    节点视为输出.
/// 4. 每帧: [`Self::draw_into_frame`].
pub struct PostProcessGraph {
	quad: FullscreenQuad,
	nodes: Vec<PostProcessNode>,

	/// 显式指定的输出节点. 为 `None` 时取最后一个节点.
	output: Option<NodeId>,

	/// 输入纹理的 bind group layout, 对应 group 1.
	input_layout: wgpu::BindGroupLayout,

	/// 构造输入 bind group 时使用的采样器.
	sampler: wgpu::Sampler,

	/// 中间目标池 —— 每个「被读取」的节点占一个槽位.
	pool: GraphTargetPool,

	/// 节点 id 到池槽位索引的映射.
	node_slots: HashMap<NodeId, usize>,

	/// 池中每个槽位的 bind group.
	slot_bind_groups: Vec<wgpu::BindGroup>,

	/// 缓存的拓扑序 —— 图结构变化时清空.
	cached_order: Option<Vec<NodeId>>,
}

impl PostProcessGraph {
	/// 创建图.
	///
	/// # 参数
	///
	/// - `input_layout` —— 输入纹理的 bind group layout, 定义每个输入
	///   的 group 布局. 通常是 binding 0 = texture, binding 1 = sampler.
	/// - `sampler` —— 输入采样器.
	pub fn new(
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		input_layout: wgpu::BindGroupLayout,
		sampler: &wgpu::Sampler,
	) -> Self {
		let quad = FullscreenQuad::new(device, queue)
			.expect("failed to create fullscreen quad");

		Self {
			quad,
			nodes: Vec::new(),
			output: None,
			input_layout,
			sampler: sampler.clone(),
			pool: GraphTargetPool::new(),
			node_slots: HashMap::new(),
			slot_bind_groups: Vec::new(),
			cached_order: None,
		}
	}

	// ================ 构建 ================

	/// 添加一个节点.
	///
	/// # 参数
	///
	/// - `render_type` —— 管线状态. 顶点格式应为
	///   [`crate::vertex::formats::position_tex`].
	/// - `shader` —— 后处理着色器.
	/// - `layout` —— 管线布局. group 0 = globals, group 1 = 主输入.
	/// - `input` —— 主输入来源.
	///
	/// # 返回
	///
	/// 节点 id —— 其他节点可以通过 [`NodeInput::Node`] 引用它.
	pub fn add_node(
		&mut self,
		render_type: Arc<RenderType>,
		shader: Shader,
		layout: LayoutHandle,
		input: NodeInput,
	) -> NodeId {
		let id = NodeId(self.nodes.len() as u32);
		self.nodes.push(PostProcessNode {
			render_type,
			shader,
			layout,
			input,
			extra_inputs: Vec::new(),
		});
		self.cached_order = None;
		id
	}

	/// 给已有节点添加一个额外输入.
	///
	/// # 参数
	///
	/// - `id` —— 目标节点.
	/// - `source` —— 输入来源.
	/// - `group` —— 绑定到哪个 group. **必须 >= 2** 且在该节点内唯一.
	///
	/// # Panics
	///
	/// - `id` 不存在.
	/// - `group < 2` —— group 0/1 已被占用.
	/// - `group` 已被该节点的另一条额外输入占用.
	pub fn add_input(&mut self, id: NodeId, source: NodeInput, group: u32) {
		assert!(group >= 2, "extra inputs must bind to group 2 or later");
		let node = self
			.nodes
			.get_mut(id.0 as usize)
			.unwrap_or_else(|| panic!("node {:?} does not exist", id));

		assert!(
			!node.extra_inputs.iter().any(|ei| ei.group == group),
			"node {:?} already has an input bound to group {}",
			id,
			group
		);

		node.extra_inputs.push(ExtraInput { source, group });
		self.cached_order = None;
	}

	/// 指定输出节点.
	///
	/// 不调用时, 最后一个添加的节点视为输出.
	///
	/// # 前置条件
	///
	/// `id` 必须指向一个已添加的节点 —— 否则 [`Self::draw_into_frame`]
	/// 返回 [`crate::pipeline::Error::InvalidNodeRef`].
	pub fn set_output(&mut self, id: NodeId) {
		self.output = Some(id);
	}

	/// 图中节点的数量.
	pub fn node_count(&self) -> usize {
		self.nodes.len()
	}

	/// 图是否不含任何节点.
	pub fn is_empty(&self) -> bool {
		self.nodes.is_empty()
	}

	/// 清空所有节点.
	///
	/// 中间目标池**不**清空 —— 保留给后续的节点复用.
	pub fn clear(&mut self) {
		self.nodes.clear();
		self.output = None;
		self.node_slots.clear();
		self.cached_order = None;
	}

	/// 图内部使用的全屏四边形.
	pub fn quad(&self) -> &FullscreenQuad {
		&self.quad
	}

	// ================ 渲染 ================

	/// 把图记录到 [`Frame`].
	///
	/// # 行为
	///
	/// 1. 拓扑排序 —— 检测环并算出执行顺序.
	/// 2. 找出需要中间目标的节点 (输出被其他节点读取的那些).
	/// 3. 按顺序记录每个节点的 pass:
	///    - 输出节点 → `final_target`.
	///    - 其他被读取的节点 → 池中的中间目标.
	///    - 既不被读取也不是输出的节点 → **跳过** (无目标可写, 也没有
	///      副作用), 见下文「死节点」.
	///
	/// # 错误
	///
	/// - 图中有环 —— [`crate::pipeline::Error::CycleDetected`].
	/// - 中间目标池重建失败 —— [`crate::pipeline::Error`].
	/// - 节点引用了不存在的 id, **或输出节点被其他节点引用**, **或
	///   输出节点 id 越界** —— [`crate::pipeline::Error::InvalidNodeRef`].
	///
	/// # 死节点
	///
	/// 既不被其他节点读取, 也不是输出节点的节点会被**完全跳过** ——
	/// 不记录任何 pass. 这些节点的输出无处可写: 用 `final_target` 当
	/// 丢弃目标会覆盖最终结果, 为它们分配临时目标又纯属浪费. 因此
	/// 死节点不产生任何副作用.
	///
	/// # Panics
	///
	/// `layout` 与 `shader` 的 group 声明不匹配时, wgpu 会在绘制时
	/// panic.
	pub fn draw_into_frame(
		&mut self,
		frame: &mut Frame<'_>,
		final_target: &impl Target,
		external_bind_group: &wgpu::BindGroup,
		device: &wgpu::Device,
	) -> Result<()> {
		if self.nodes.is_empty() {
			return Ok(());
		}

		let order = self.topological_order()?;
		let output = self
			.output
			.unwrap_or_else(|| *order.last().expect("non-empty"));

		// 显式设置的输出 id 必须指向一个存在的节点.
		//
		// **不要**省略这一步: 若调用方 `set_output` 时传了越界的 id,
		// 下面的 `read_nodes` 与循环都只在 0..n 内匹配 id, 因此都
		// 不会命中它, 结果是什么都不写进 `final_target` —— 用户看到
		// 的是「黑屏但无错误」. 在这里显式报错, 让问题在开发阶段就
		// 暴露.
		//
		// 注: 未设置 `output` 时取拓扑序的最后一个节点, 该 id 一定在
		// 范围内, 不会触发此错误.
		if output.0 as usize >= self.nodes.len() {
			return Err(crate::pipeline::Error::InvalidNodeRef { id: output.0 });
		}

		// 收集所有「被其他节点读取」的节点 —— 它们需要中间目标.
		let mut read_nodes: Vec<NodeId> = Vec::new();
		for node in &self.nodes {
			collect_read_ids(node, &mut read_nodes);
		}

		// 输出节点不允许被其他节点引用 —— 否则它的结果去向有二义性
		// (既写 `final_target` 又要作为中间输入). 这里显式拒绝, 以免
		// `resolve_input` 在运行期 panic.
		if read_nodes.contains(&output) {
			return Err(crate::pipeline::Error::InvalidNodeRef { id: output.0 });
		}

		// 输出节点不需要中间目标 —— 把它从读取集合里剔除.
		read_nodes.retain(|id| *id != output);
		read_nodes.sort();
		read_nodes.dedup();

		let (width, height) = final_target.viewport_size();
		let color_format = final_target.color_format();

		let pool_changed = self.pool.ensure(
			device,
			&read_nodes,
			width,
			height,
			color_format,
		)?;
		if pool_changed {
			self.rebuild_slot_bind_groups(device);
		}

		// 建立节点 id 到池槽位的映射.
		self.node_slots.clear();
		for (slot, id) in read_nodes.iter().enumerate() {
			self.node_slots.insert(*id, slot);
		}

		for id in order {
			let node = &self.nodes[id.0 as usize];

			// 解析主输入的 bind group.
			let primary = self.resolve_input(node.input, external_bind_group);

			// 逐个解析额外输入.
			let mut extra: Vec<(u32, &wgpu::BindGroup)> = Vec::new();
			for ei in &node.extra_inputs {
				let bg = self.resolve_input(ei.source, external_bind_group);
				extra.push((ei.group, bg));
			}

			let quad_mesh = self.quad.mesh();

			if id == output {
				frame.pass(final_target, |pass_helper| {
					pass_helper.set_render_type(
						&node.render_type,
						&node.shader,
						&node.layout,
					);
					pass_helper.set_bind_group(1, primary, &[]);
					for (group, bg) in &extra {
						pass_helper.set_bind_group(*group, bg, &[]);
					}
					pass_helper.draw_mesh(quad_mesh);
				});
			} else if let Some(&slot) = self.node_slots.get(&id) {
				let target = &self.pool.targets[slot];
				frame.pass(target, |pass_helper| {
					pass_helper.set_render_type(
						&node.render_type,
						&node.shader,
						&node.layout,
					);
					pass_helper.set_bind_group(1, primary, &[]);
					for (group, bg) in &extra {
						pass_helper.set_bind_group(*group, bg, &[]);
					}
					pass_helper.draw_mesh(quad_mesh);
				});
			} else {
				// 不被读取且不是输出 —— 无目标可写, 也没有副作用.
				//
				// 把 `final_target` 当作丢弃目标会覆盖最终结果; 为它
				// 分配一个中间目标又纯属浪费. 跳过记录是唯一合理的
				// 处理.
				//
				// **不要**改成写入临时目标: 那会让「死节点」意外消耗
				// 池槽位, 挤占真正需要的中间目标.
			}
		}

		Ok(())
	}

	/// 把 [`NodeInput`] 解析成可直接绑定的 bind group.
	fn resolve_input<'a>(
		&'a self,
		input: NodeInput,
		external: &'a wgpu::BindGroup,
	) -> &'a wgpu::BindGroup {
		match input {
			NodeInput::External => external,
			NodeInput::Node(id) => {
				let slot = self
					.node_slots
					.get(&id)
					.unwrap_or_else(|| panic!("node {:?} has no intermediate target", id));
				&self.slot_bind_groups[*slot]
			}
		}
	}

	// ================ 拓扑排序 ================

	/// 计算拓扑序 —— 依赖在前.
	///
	/// # 错误
	///
	/// - 引用了不存在的节点 —— [`crate::pipeline::Error::InvalidNodeRef`].
	/// - 图中有环 —— [`crate::pipeline::Error::CycleDetected`].
	///
	/// # 与输出节点
	///
	/// 本方法**不**校验 [`Self::output`] —— 输出 id 的越界检查放在
	/// [`Self::draw_into_frame`] 里做: 拓扑序可以被缓存, 而输出 id
	/// 的验证不应被缓存跳过.
	fn topological_order(&mut self) -> Result<Vec<NodeId>> {
		if let Some(order) = &self.cached_order {
			return Ok(order.clone());
		}

		let n = self.nodes.len();

		// 先校验所有引用到的节点 id 都落在 0..n 内.
		for node in &self.nodes {
			for id in node_dependencies(node) {
				if id.0 as usize >= n {
					return Err(crate::pipeline::Error::InvalidNodeRef {
						id: id.0,
					});
				}
			}
		}

		// Kahn 算法.
		// 邻接表 —— 从「被依赖的节点」指向「依赖它的节点」.
		let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];
		let mut in_degree: Vec<u32> = vec![0; n];

		for (i, node) in self.nodes.iter().enumerate() {
			for dep in node_dependencies(node) {
				dependents[dep.0 as usize].push(i);
				in_degree[i] += 1;
			}
		}

		// 入度为 0 的节点没有依赖, 可以最先执行.
		let mut queue: std::collections::VecDeque<usize> =
			(0..n).filter(|&i| in_degree[i] == 0).collect();

		let mut order = Vec::with_capacity(n);
		while let Some(i) = queue.pop_front() {
			order.push(NodeId(i as u32));
			for &d in &dependents[i] {
				in_degree[d] -= 1;
				if in_degree[d] == 0 {
					queue.push_back(d);
				}
			}
		}

		if order.len() != n {
			return Err(crate::pipeline::Error::CycleDetected);
		}

		self.cached_order = Some(order.clone());
		Ok(order)
	}

	/// 为池中每个槽位重建 bind group.
	fn rebuild_slot_bind_groups(&mut self, device: &wgpu::Device) {
		self.slot_bind_groups.clear();
		self.slot_bind_groups.reserve(self.pool.targets.len());

		for target in &self.pool.targets {
			let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some("post_graph_slot_bind_group"),
				layout: &self.input_layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: wgpu::BindingResource::TextureView(&target.color_view),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: wgpu::BindingResource::Sampler(&self.sampler),
					},
				],
			});
			self.slot_bind_groups.push(bg);
		}
	}
}

// ================ 中间目标池 ================

/// 图中间目标池.
///
/// # 与线性链的差异
///
/// 线性链用 ping-pong 策略固定 2 个目标; 图的目标数取决于「被读取的
/// 节点数」—— 池按需扩展, 不收缩.
struct GraphTargetPool {
	targets: Vec<RenderTarget>,
	width: u32,
	height: u32,
	color_format: ColorFormat,

	/// 当前池对应的节点 id 列表 —— 顺序一变就要重建 bind group.
	layout: Vec<NodeId>,
}

impl GraphTargetPool {
	fn new() -> Self {
		Self {
			targets: Vec::new(),
			width: 0,
			height: 0,
			color_format: ColorFormat::Rgba8Unorm,
			layout: Vec::new(),
		}
	}

	/// 让池与 `read_nodes` 一一对应.
	///
	/// # 返回值
	///
	/// `true` 表示池发生了变化 —— 调用方需要重建 bind group.
	fn ensure(
		&mut self,
		device: &wgpu::Device,
		read_nodes: &[NodeId],
		width: u32,
		height: u32,
		color_format: ColorFormat,
	) -> Result<bool> {
		let mut changed = false;

		// 尺寸或格式变化 —— 全部重建.
		if self.width != width
			|| self.height != height
			|| self.color_format != color_format
		{
			self.targets.clear();
			self.layout.clear();
			self.width = width;
			self.height = height;
			self.color_format = color_format;
			changed = true;
		}

		// 节点集合变化 —— 重建映射, 目标数量按需扩展.
		if self.layout != read_nodes {
			self.layout = read_nodes.to_vec();
			changed = true;
		}

		while self.targets.len() < read_nodes.len() {
			let target = RenderTarget::new(
				device,
				width,
				height,
				color_format,
				None,
				Some("post_graph_intermediate"),
			)?;
			self.targets.push(target);
			changed = true;
		}

		Ok(changed)
	}
}

// ================ 辅助 ================

/// 收集一个节点依赖的全部节点 id (主输入 + 额外输入).
fn collect_read_ids(node: &PostProcessNode, out: &mut Vec<NodeId>) {
	for id in node_dependencies(node) {
		out.push(id);
	}
}

/// 逐个产出一个节点依赖的节点 id.
fn node_dependencies(node: &PostProcessNode) -> impl Iterator<Item = NodeId> + '_ {
	let primary = match node.input {
		NodeInput::External => None,
		NodeInput::Node(id) => Some(id),
	};
	primary
		.into_iter()
		.chain(node.extra_inputs.iter().filter_map(|ei| match ei.source {
			NodeInput::External => None,
			NodeInput::Node(id) => Some(id),
		}))
}

/// 把 [`PostProcessInput`] 转换成 [`NodeInput`].
///
/// 便于把线性链的 pass 迁移到图. `Previous` 需要调用方替换成具体的
/// 节点 id, 因此本函数对 `Previous` 返回 `None`.
pub fn input_from_chain(input: PostProcessInput) -> Option<NodeInput> {
	match input {
		PostProcessInput::External => Some(NodeInput::External),
		PostProcessInput::Previous => None,
	}
}
