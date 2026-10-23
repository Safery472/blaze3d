//! 渲染列表 —— 收集绘制项, 排序后批量绘制.
//!
//! [`RenderList`] 先收集一组 [`DrawItem`] —— 网格, 对象 uniform 偏移,
//! 管线状态与绑定组 —— 再按调用方给出的顺序排序, 最后批量绘制.
//!
//! # 排序
//!
//! [`RenderList::sort`] 依次按下列键比较:
//!
//! 1. [`DrawItem::layer`] 升序 —— 由调用方决定谁先绘制, 默认 0.
//! 2. 管线哈希升序 —— 同一管线的绘制项被排到一起, 减少状态切换.
//! 3. [`DrawItem::depth`] 降序 —— 远的先画, 与 alpha 混合的预期一致.
//! 4. 原始插入顺序 —— 排序稳定, 结果可预测.
//!
//! 管线哈希由 `(render_type, shader.id, layout.id())` 派生. 哈希冲突
//! 会让两个无关的管线落进同一组 —— 只影响排序质量, 不影响正确性,
//! 因为真正绘制时仍会重新绑定管线.
//!
//! # 剔除
//!
//! [`RenderList::retain_in_frustum`] 依据 [`DrawItem::bounds`] 剔除
//! 视锥外的绘制项. `bounds` 为 `None` 的项会被保守保留 —— 没有包围盒
//! 就不做剔除.
//!
//! # 与 `RenderSystem` 的关系
//!
//! 渲染列表是**用户层的收集结构**, 并不持有资源. 具体来说它:
//!
//! - 不持有 wgpu 资源, 只保留 `Arc` 句柄.
//! - 不驱动渲染循环 —— 那由 [`RenderSystem::render_frame`] 负责.
//! - 不管理对象 uniform 缓冲 —— 那由 [`RenderSystem::push_object`]
//!   负责.
//!
//! 典型用法如下:
//!
//! ```ignore
//! let mut list = RenderList::new();
//!
//! // 收集本帧的绘制项.
//! for obj in visible_objects {
//!     let offset = system.push_object(obj.transform)?;
//!     list.push(
//!         DrawItem::new(obj.mesh.clone(), offset, rt.clone(), shader.clone(), layout.clone())
//!             .with_depth(distance_to_camera(obj.transform))
//!             .with_bounds(obj.aabb)
//!     );
//! }
//!
//! // 剔除.
//! list.retain_in_frustum(&camera.frustum());
//!
//! // 渲染.
//! system.render_frame(&target, |pass| {
//!     list.draw(pass);
//! });
//! ```
//!
//! [`RenderSystem::render_frame`]: crate::system::RenderSystem::render_frame
//! [`RenderSystem::push_object`]: crate::system::RenderSystem::push_object

use std::sync::Arc;

use crate::camera::{Aabb, Frustum};
use crate::shader::Shader;
use crate::state::{LayoutHandle, RenderType};
use crate::system::{PassHelper, SharedGpuMesh};

/// 渲染列表中的一个绘制项.
///
/// # 字段
///
/// 字段全部公开. 建议用 [`Self::new`] 配合构建器方法构造 —— 直接写
/// 结构体字面量就得自己了解管线哈希的内部细节.
///
/// # `bind_groups`
///
/// 这里放 group 2 及以后的绑定. group 0 (globals) 与 group 1
/// (对象 uniform) 由 [`PassHelper`] 自动绑定, **不要**写进来.
///
/// 按 group 索引升序存储可以略省查找开销, 但并不强制.
pub struct DrawItem {
	/// 本项要绘制的网格.
	pub mesh: SharedGpuMesh,

	/// 对象 uniform 在缓冲中的偏移.
	///
	/// 取自 [`crate::system::RenderSystem::push_object`]; 绘制时
	/// 传给 [`PassHelper::set_object`].
	pub object_offset: u32,

	/// 本条绘制使用的管线状态.
	///
	/// 包一层 `Arc`, 好让同一个 `RenderType` 在多个绘制项之间共享 ——
	/// `RenderType` 内部含 `VertexFormat`, 克隆代价不低.
	pub render_type: Arc<RenderType>,

	/// 本项使用的着色器.
	pub shader: Shader,

	/// 本项使用的管线布局.
	pub layout: LayoutHandle,

	/// 附加的绑定组 (group 2 及以后).
	pub bind_groups: Vec<(u32, Arc<wgpu::BindGroup>)>,

	/// 排序用的层号.
	///
	/// 值小的先绘制, 默认为 0. 用非零值可把绘制项推到队列前面或
	/// 后面.
	pub layer: u32,

	/// 排序用的深度键.
	///
	/// **降序** —— 值大的先绘制, 供半透明物体实现「远的先画」.
	/// 对不透明物体而言顺序无关紧要, 深度测试会处理.
	pub depth: f32,

	/// 世界空间的包围盒.
	///
	/// 供 [`RenderList::retain_in_frustum`] 剔除时使用. `None` 表示
	/// **不参与剔除**, 即保守保留. 没有包围盒可言的场景 (全屏四边形,
	/// 调试线) 应当留 `None`.
	pub bounds: Option<Aabb>,
}

impl DrawItem {
	/// 构造一个绘制项.
	///
	/// 各可选项的初值:
	///
	/// - `bind_groups` —— 空.
	/// - `layer` —— 0.
	/// - `depth` —— 0.0.
	/// - `bounds` —— `None`, 即不参与剔除.
	pub fn new(
		mesh: SharedGpuMesh,
		object_offset: u32,
		render_type: Arc<RenderType>,
		shader: Shader,
		layout: LayoutHandle,
	) -> Self {
		Self {
			mesh,
			object_offset,
			render_type,
			shader,
			layout,
			bind_groups: Vec::new(),
			layer: 0,
			depth: 0.0,
			bounds: None,
		}
	}

	/// 追加一个绑定组, 供 group 2 及以后使用.
	pub fn with_bind_group(
		mut self,
		group: u32,
		bind_group: Arc<wgpu::BindGroup>,
	) -> Self {
		self.bind_groups.push((group, bind_group));
		self
	}

	/// 指定排序层 —— 值小的先绘制.
	pub fn with_layer(mut self, layer: u32) -> Self {
		self.layer = layer;
		self
	}

	/// 指定深度排序键 —— 值大的先绘制.
	pub fn with_depth(mut self, depth: f32) -> Self {
		self.depth = depth;
		self
	}

	/// 指定世界空间包围盒.
	///
	/// 设置之后, 本项才会参与 [`RenderList::retain_in_frustum`] 的
	/// 剔除.
	pub fn with_bounds(mut self, bounds: Aabb) -> Self {
		self.bounds = Some(bounds);
		self
	}
}

/// 绘制项的收集与排序容器.
///
/// # 排序
///
/// 排序是**惰性**的: 首次 [`Self::draw`] 时若 `sorted == false`, 会
/// 自动调用 [`Self::sort`]. 提前显式调用 `sort` 就能免去绘制时的这次
/// 排序.
///
/// # 剔除
///
/// [`Self::retain_in_frustum`] 就地删掉视锥外的项, 与 [`Self::push`]
/// 一样会把排序状态置为未排序 —— 下次 `draw` 会重新排一遍.
///
/// 调用位置应在 `push` 全部完成之后, `draw` 之前.
///
/// # 复用
///
/// [`Self::clear`] 保留内部 `Vec` 的容量, 因此可以跨帧复用. 每帧开头
/// 先 `clear` 是个好习惯.
pub struct RenderList {
	items: Vec<DrawItem>,

	/// 排好序的索引顺序.
	///
	/// 为空表示还没排序, 首次 `draw` 时才会填充.
	order: Vec<u32>,

	/// 排序状态标志.
	///
	/// `push` 与 `retain_in_frustum` 把它置为 false, `sort` 置为 true.
	sorted: bool,
}

impl Default for RenderList {
	fn default() -> Self {
		Self::new()
	}
}

impl RenderList {
	/// 创建一个空渲染列表.
	pub const fn new() -> Self {
		Self {
			items: Vec::new(),
			order: Vec::new(),
			sorted: true,
		}
	}

	/// 追加一个绘制项.
	///
	/// 追加后列表被标记为「未排序」, 下次 `draw` 会重新排序.
	pub fn push(&mut self, item: DrawItem) {
		self.items.push(item);
		self.sorted = false;
	}

	/// 清空全部绘制项.
	///
	/// 内部 `Vec` 的容量会保留, 以便后续复用.
	pub fn clear(&mut self) {
		self.items.clear();
		self.order.clear();
		self.sorted = true;
	}

	/// 返回当前的绘制项数量.
	pub fn len(&self) -> usize {
		self.items.len()
	}

	/// 返回列表是否为空.
	pub fn is_empty(&self) -> bool {
		self.items.is_empty()
	}

	/// 剔除落在视锥之外的绘制项.
	///
	/// # 行为
	///
	/// 逐个检查: [`DrawItem::bounds`] 为 `Some` 且与视锥不相交的项会
	/// 被删掉; `bounds` 为 `None` 的项则**保守保留** —— 没有包围盒就
	/// 不做剔除.
	///
	/// 只要有项被删, 列表就标记为「未排序」, 下次 `draw` 会重新排序.
	///
	/// # 保守性
	///
	/// [`Aabb::intersects_frustum`] 把角落恰好贴住视锥的情形判为相交
	/// —— 宁可多画, 不可少画. 因此剔除不会让可见物体凭空消失.
	///
	/// # 调用时机
	///
	/// - 收集完成之后, 绘制之前. 典型流程是多次 `push`, 接着
	///   `retain_in_frustum`, 最后 `draw`.
	/// - 可以重复调用: 换不同视锥多次剔除是合法的, 每次都在上一次
	///   的结果上继续过滤.
	///
	/// # 与 sort 的先后
	///
	/// 两者谁先谁后都可以. 但 `retain_in_frustum` 会把 `sorted` 重置,
	/// 所以**先剔除再排序**更省 —— 参与排序的数据更少.
	///
	/// [`Aabb::intersects_frustum`]: crate::camera::Aabb::intersects_frustum
	pub fn retain_in_frustum(&mut self, frustum: &Frustum) {
		let before = self.items.len();
		self.items.retain(|item| match &item.bounds {
			Some(aabb) => aabb.intersects_frustum(frustum),
			// 没有包围盒 —— 保守保留, 见 `DrawItem::bounds` 的文档.
			None => true,
		});

		if self.items.len() != before {
			self.sorted = false;
		}
	}

	/// 按模块文档给出的规则排序.
	///
	/// 排序是稳定的 —— `(layer, pipeline, depth)` 完全相同的项仍保持
	/// 原始的插入顺序.
	pub fn sort(&mut self) {
		// 先把管线哈希算好, 每项只算一次, 比较器里就不必重复算.
		let hashes: Vec<u64> = self.items.iter().map(pipeline_hash).collect();

		self.order.clear();
		self.order.extend(0..self.items.len() as u32);

		let items = &self.items;
		self.order.sort_by(|&a, &b| {
			let ia = &items[a as usize];
			let ib = &items[b as usize];

			ia.layer
				.cmp(&ib.layer)
				.then_with(|| hashes[a as usize].cmp(&hashes[b as usize]))
				.then_with(|| {
					// 降序排列: 远的先画; NaN 一律按相等处理.
					ib.depth
						.partial_cmp(&ia.depth)
						.unwrap_or(std::cmp::Ordering::Equal)
				})
				.then_with(|| a.cmp(&b))
		});

		self.sorted = true;
	}

	/// 按排序结果绘制全部项.
	///
	/// # 行为
	///
	/// 列表尚未排序时先调用 [`Self::sort`], 然后按排序后的顺序逐项:
	///
	/// 1. 用 [`PassHelper::set_render_type`] 设置管线, 内部会去重.
	/// 2. 用 [`PassHelper::set_object`] 绑定对象 uniform.
	/// 3. 逐个绑定 `bind_groups` 里的组.
	/// 4. 用 [`PassHelper::draw_mesh`] 绘制网格.
	///
	/// # 状态去重
	///
	/// `PassHelper` 自己记着当前管线, 连续绘制同一管线时不会重复
	/// `set_pipeline`. 于是排序之后, 管线切换次数等于「不同管线的连续
	/// 段数」, 通常远小于绘制项总数.
	pub fn draw(&mut self, pass: &mut PassHelper<'_, '_>) {
		if !self.sorted {
			self.sort();
		}

		for &idx in &self.order {
			let item = &self.items[idx as usize];

			pass.set_render_type(&item.render_type, &item.shader, &item.layout);
			pass.set_object(item.object_offset);

			for (group, bind_group) in &item.bind_groups {
				pass.set_bind_group(*group, bind_group, &[]);
			}

			pass.draw_mesh(&item.mesh);
		}
	}
}

/// 算出绘制项对应的管线哈希.
///
/// 输入为 `(render_type, shader.id, layout.id())` —— 决定一条管线的
/// 三个维度. 哈希相同的绘制项会被 [`RenderList::sort`] 排到一起.
fn pipeline_hash(item: &DrawItem) -> u64 {
	use std::hash::{Hash, Hasher};

	let mut hasher = std::collections::hash_map::DefaultHasher::new();
	item.render_type.hash(&mut hasher);
	item.shader.id.hash(&mut hasher);
	item.layout.id().hash(&mut hasher);
	hasher.finish()
}
