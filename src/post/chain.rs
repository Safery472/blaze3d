// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 后处理链.
//!
//! [`PostProcessChain`] 管理一串后处理 pass: 每个 pass 都用全屏四边形
//! 采样输入纹理, 应用效果后写入下一个目标.
//!
//! # 数据流
//!
//! 链是**线性**的 —— pass 按 `add_pass` 的顺序依次执行. 每个 pass 的
//! 输入只有两种选择:
//!
//! - [`PostProcessInput::External`] —— `draw_into_frame` 时传入的 bind
//!   group. 第一个 pass 通常用这个.
//! - [`PostProcessInput::Previous`] —— 上一个 pass 的输出.
//!
//! 最后一个 pass 输出到 `draw_into_frame` 的 `final_target`; 它之前的
//! pass 输出到链内部的中间目标.
//!
//! # 中间目标池
//!
//! pass 之间的中间结果写入**目标池**. 池按尺寸与格式自动重建, 并可
//! 跨帧复用 —— 尺寸未变时不会重新分配.
//!
//! 池使用 **ping-pong** 策略: 只需 2 个目标就能支持任意长度的链.
//!
//! ```text
//! N = 4 个 pass:
//!   pass 0: external → slot 0
//!   pass 1: slot 0 → slot 1
//!   pass 2: slot 1 → slot 0
//!   pass 3: slot 0 → final_target
//! ```
//!
//! 每个 pass 的输出槽位是 `pass_index % 2` —— 与它读的槽位必然不同
//! (pass N 读 `(N-1) % 2`, 写 `N % 2`).
//!
//! # Bind group 约定
//!
//! 后处理 pass 使用的管线布局必须遵循下面的约定:
//!
//! - **group 0** —— globals (帧级 uniform). 由
//!   [`crate::system::PassHelper::set_render_type`] 自动绑定.
//! - **group 1** —— 输入纹理 (texture + sampler). 由本模块绑定.
//! - **group 2+** —— 用户自定义, 在 pass 回调里手动绑定.
//!
//! 输入纹理的 bind group layout 在构造时由 `input_layout` 参数提供,
//! 它定义 group 1 的布局 (通常是 binding 0 = texture, binding 1 =
//! sampler).
//!
//! # 与 `RenderSystem` 的关系
//!
//! [`PostProcessChain`] 不持有 [`crate::system::RenderSystem`] —— 它把
//! 绘制命令记录到调用方提供的 [`crate::system::Frame`] 上. 因此 chain
//! 的所有 pass 与调用方自己的 pass 共用同一个 command encoder, 只需
//! 一次提交.

use std::sync::Arc;

use crate::pipeline::{RenderTarget, Result, Target};
use crate::shader::Shader;
use crate::state::{ColorFormat, LayoutHandle, RenderType};
use crate::system::Frame;

use super::quad::FullscreenQuad;

// ================ 输入来源 ================

/// 后处理 pass 的输入来源.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PostProcessInput {
	/// 外部输入.
	///
	/// 即 [`PostProcessChain::draw_into_frame`] 调用时传入的
	/// `external_bind_group`. 第一个 pass 通常用它读取场景纹理.
	External,

	/// 上一个 pass 的输出.
	///
	/// 第一个 pass 使用它会 panic —— 它没有「上一个 pass」.
	Previous,
}

// ================ Pass ================

/// 单个后处理 pass.
///
/// 由 [`PostProcessChain::add_pass`] 创建. 字段全部公开 —— 链的内容
/// 不维护额外的不变量.
pub struct PostProcessPass {
	/// 管线状态.
	///
	/// 顶点格式应与全屏四边形匹配, 即
	/// [`crate::vertex::formats::position_tex`].
	pub render_type: Arc<RenderType>,

	/// 该 pass 使用的后处理着色器.
	pub shader: Shader,

	/// 管线布局.
	///
	/// **必须**遵循 group 0 = globals, group 1 = 输入纹理的约定. 见
	/// 模块文档的「Bind group 约定」.
	pub layout: LayoutHandle,

	/// 该 pass 的输入来源.
	pub input: PostProcessInput,
}

// ================ 链 ================

/// 后处理链.
///
/// # 使用流程
///
/// 1. 启动阶段: [`Self::new`] 创建链, 再用 [`Self::add_pass`] 逐个
///    添加 pass.
/// 2. 每帧: [`Self::draw_into_frame`] 把 pass 记录到 [`Frame`].
///
/// # 尺寸变化
///
/// 链本身不跟踪尺寸 —— `draw_into_frame` 时按 `final_target` 的尺寸
/// 自动调整中间目标池.
pub struct PostProcessChain {
	quad: FullscreenQuad,
	passes: Vec<PostProcessPass>,

	/// 输入纹理的 bind group layout.
	///
	/// 定义 group 1 的布局. 每个 pass 的 `layout` 中 group 1 必须与
	/// 之一致 —— 否则绘制时 wgpu 会 panic.
	input_layout: wgpu::BindGroupLayout,

	/// 输入采样器.
	///
	/// 对应 `input_layout` 里的 sampler binding.
	sampler: wgpu::Sampler,

	/// 存放 pass 之间中间结果的目标池.
	pool: IntermediatePool,

	/// 每个池槽位的 bind group.
	///
	/// `slot_bind_groups[i]` 绑定 `pool.targets[i].color_view`, 作为
	/// 下一个 pass 的 group 1 输入.
	///
	/// 池重建时同步重建.
	slot_bind_groups: Vec<wgpu::BindGroup>,
}

impl PostProcessChain {
	/// 创建链.
	///
	/// # 参数
	///
	/// - `device` —— 用于创建全屏四边形.
	/// - `queue` —— 用于上传四边形的顶点数据.
	/// - `input_layout` —— 输入纹理的 bind group layout, 定义每个 pass
	///   的 group 1 布局. 通常是 binding 0 = texture, binding 1 =
	///   sampler.
	/// - `sampler` —— 输入采样器, 需与 `input_layout` 匹配.
	///
	/// # Panics
	///
	/// 全屏四边形创建失败时 panic. 这属于编程错误 —— 4 个顶点的四边形
	/// 不应失败.
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
			passes: Vec::new(),
			input_layout,
			sampler: sampler.clone(),
			pool: IntermediatePool::new(),
			slot_bind_groups: Vec::new(),
		}
	}

	/// 添加一个 pass.
	///
	/// # 顺序
	///
	/// pass 按添加顺序执行. 第一个 pass 的 `input` 通常用
	/// [`PostProcessInput::External`] —— 它没有「上一个 pass」.
	///
	/// # 参数
	///
	/// - `render_type` —— 管线状态. 顶点格式应为
	///   [`crate::vertex::formats::position_tex`].
	/// - `shader` —— 后处理着色器.
	/// - `layout` —— 管线布局. 必须遵循 group 0 = globals, group 1 =
	///   输入纹理的约定.
	/// - `input` —— 输入来源.
	pub fn add_pass(
		&mut self,
		render_type: Arc<RenderType>,
		shader: Shader,
		layout: LayoutHandle,
		input: PostProcessInput,
	) {
		self.passes.push(PostProcessPass {
			render_type,
			shader,
			layout,
			input,
		});
	}

	/// 清空所有 pass.
	///
	/// 中间目标池**不**清空 —— 保留给后续的 pass 复用.
	pub fn clear_passes(&mut self) {
		self.passes.clear();
	}

	/// 链内部使用的全屏四边形.
	pub fn quad(&self) -> &FullscreenQuad {
		&self.quad
	}

	/// 链中 pass 的数量.
	pub fn pass_count(&self) -> usize {
		self.passes.len()
	}

	/// 链是否不含任何 pass.
	pub fn is_empty(&self) -> bool {
		self.passes.is_empty()
	}

	/// 把后处理链记录到 [`Frame`].
	///
	/// # 行为
	///
	/// 按 `add_pass` 的顺序记录每个 pass. 最后一个 pass 输出到
	/// `final_target`, 其余的 pass 输出到内部的中间目标.
	///
	/// 中间目标池按 `final_target` 的尺寸与颜色格式自动调整 —— 尺寸
	/// 或格式变化时重建, 数量不足时扩展.
	///
	/// # 无 pass
	///
	/// 链为空时本方法是 no-op, 场景内容不会被复制到 `final_target`.
	/// 需要「直接把场景复制到窗口」时, 加一个不做任何效果的 pass
	/// (着色器直接输出采样值).
	///
	/// # 错误
	///
	/// 中间目标池重建失败时返回 [`crate::pipeline::Error`]. 通常只在
	/// 设备丢失或尺寸超限时发生.
	///
	/// # Panics
	///
	/// 第一个 pass 的 `input` 是 [`PostProcessInput::Previous`] 时
	/// panic —— 它没有「上一个 pass」.
	pub fn draw_into_frame(
		&mut self,
		frame: &mut Frame<'_>,
		final_target: &impl Target,
		external_bind_group: &wgpu::BindGroup,
		device: &wgpu::Device,
	) -> Result<()> {
		if self.passes.is_empty() {
			return Ok(());
		}

		// 第一个 pass 不能取「上一个 pass」的输出.
		if matches!(self.passes[0].input, PostProcessInput::Previous) {
			panic!("first pass cannot use PostProcessInput::Previous");
		}

		let (width, height) = final_target.viewport_size();
		let color_format = final_target.color_format();

		// 需要几个中间目标:
		//
		//   N = 1 —— 0 个 (external 直接写 final).
		//   N = 2 —— 1 个 (pass 0 写 slot 0, pass 1 读 slot 0 后写最终目标).
		//   N >= 3 —— 2 个 (ping-pong).
		let needed = 2.min(self.passes.len().saturating_sub(1));

		let pool_changed = self
			.pool
			.ensure(device, needed, width, height, color_format)?;
		if pool_changed {
			self.rebuild_bind_groups(device);
		}

		let pass_count = self.passes.len();

		for i in 0..pass_count {
			let is_last = i == pass_count - 1;
			let slot = i % 2;

			// 当前 pass 的输入 bind group.
			let input_bg: &wgpu::BindGroup = match self.passes[i].input {
				PostProcessInput::External => external_bind_group,
				PostProcessInput::Previous => {
					// 第一个 pass 已在前面校验过, 因此这里 i > 0 恒成立.
					let prev_slot = (i - 1) % 2;
					&self.slot_bind_groups[prev_slot]
				}
			};

			// 借用拆分 —— `pass_def`, `quad_mesh` 与 `target` 都是
			// `self` 的不可变借用, 与 `frame` 的可变借用互不冲突.
			let pass_def = &self.passes[i];
			let quad_mesh = self.quad.mesh();

			if is_last {
				frame.pass(final_target, |pass_helper| {
					pass_helper.set_render_type(
						&pass_def.render_type,
						&pass_def.shader,
						&pass_def.layout,
					);
					// group 0 由 `set_render_type` 自动绑定 (globals).
					// group 1 则是本模块绑定的输入纹理.
					pass_helper.set_bind_group(1, input_bg, &[]);
					pass_helper.draw_mesh(quad_mesh);
				});
			} else {
				let target = &self.pool.targets[slot];
				frame.pass(target, |pass_helper| {
					pass_helper.set_render_type(
						&pass_def.render_type,
						&pass_def.shader,
						&pass_def.layout,
					);
					pass_helper.set_bind_group(1, input_bg, &[]);
					pass_helper.draw_mesh(quad_mesh);
				});
			}
		}

		Ok(())
	}

	/// 为所有槽位重建 bind group.
	///
	/// # 调用时机
	///
	/// 中间目标池重建之后 —— 旧的 `color_view` 已全部失效.
	///
	/// # 为什么 bind group 不放在目标池里
	///
	/// [`IntermediatePool`] 只持有 [`crate::pipeline::RenderTarget`], 不
	/// 知道输入 layout 与采样器 —— 这两者是 chain 的状态. 创建 bind
	/// group 需要两者结合, 因此放在 chain 里.
	fn rebuild_bind_groups(&mut self, device: &wgpu::Device) {
		self.slot_bind_groups.clear();
		self.slot_bind_groups.reserve(self.pool.targets.len());

		for target in &self.pool.targets {
			let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some("post_slot_bind_group"),
				layout: &self.input_layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: wgpu::BindingResource::TextureView(
							&target.color_view,
						),
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

/// 中间目标池.
///
/// # 重建语义
///
/// 尺寸或格式变化时清空重建; 数量不足时扩展, 数量超出时不收缩 ——
/// 后续的 draw 可能还需要.
///
/// # 跨帧复用
///
/// 池的生命周期跨帧 —— 尺寸未变时不会重新分配. 这与
/// [`crate::pipeline::RenderTarget`] 的设计目标 (长期持有) 一致.
struct IntermediatePool {
	targets: Vec<RenderTarget>,
	width: u32,
	height: u32,
	color_format: ColorFormat,
}

impl IntermediatePool {
	fn new() -> Self {
		Self {
			targets: Vec::new(),
			width: 0,
			height: 0,
			color_format: ColorFormat::Rgba8Unorm,
		}
	}

	/// 确保池至少有 `count` 个目标, 且尺寸与格式匹配.
	///
	/// # 返回值
	///
	/// `true` 表示池发生了变化 —— 调用方需要重建依赖
	/// `targets[i].color_view` 的资源 (例如 bind group).
	///
	/// # 尺寸 / 格式变化的处理
	///
	/// 清空所有目标后重建, 旧目标的 `color_view` 全部失效.
	fn ensure(
		&mut self,
		device: &wgpu::Device,
		count: usize,
		width: u32,
		height: u32,
		color_format: ColorFormat,
	) -> Result<bool> {
		let mut changed = false;

		if self.width != width
			|| self.height != height
			|| self.color_format != color_format
		{
			self.targets.clear();
			self.width = width;
			self.height = height;
			self.color_format = color_format;
			changed = true;
		}

		while self.targets.len() < count {
			let target = RenderTarget::new(
				device,
				width,
				height,
				color_format,
				None,
				Some("post_intermediate"),
			)?;
			self.targets.push(target);
			changed = true;
		}

		Ok(changed)
	}
}
