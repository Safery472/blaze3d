//! 带唯一 `id` 的管线布局句柄.
//!
//! [`LayoutHandle`] 把 [`wgpu::PipelineLayout`] 与它的唯一 `id` 绑定在一起.
//! 调用方注册一次, 之后把 handle 传给渲染接口即可 —— 既不需要手工维护
//! `id`, 也不会出现「`id` 与 `layout` 不匹配」的错误.
//!
//! # 为什么用 handle 而不是「`layout` 缓存 + 名字查找」
//!
//! 备选方案是在 [`RenderSystem`][] 里维护一张 `name -> layout` 表, 调用
//! 时按名字查找. 其缺点:
//!
//! - 字符串比较存在开销 (虽然很小).
//! - 名字拼错时只能在运行时发现.
//! - 调用方必须记住「哪个名字对应哪个组合」.
//!
//! handle 方案则:
//!
//! - 无需任何查找, 直接拿到 `Arc<wgpu::PipelineLayout>`.
//! - 类型系统保证 handle 一定已被注册.
//! - `id` 自动分配, 无需手工管理.
//!
//! 唯一的代价是 handle 需要被持有. 但它实现了 `Clone` 且内部是 `Arc`,
//! 拷贝的成本极低.
//!
//! # 生命周期
//!
//! 底层 `wgpu::PipelineLayout` 用 `Arc` 包装 —— 只要还有 handle 引用就
//! 保持有效. handle 全部 drop 后, wgpu 对象随之释放.
//!
//! [`RenderSystem`]: crate::system::RenderSystem

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// 布局 `id` 的全局分配器.
///
/// 从 1 开始递增 —— 0 保留给未初始化状态. 用 `Relaxed` 序: 它只是
/// 唯一性标识, 不参与同步, 也不需要更强的内存序.
static NEXT_LAYOUT_ID: AtomicU64 = AtomicU64::new(1);

/// 管线布局与它的唯一 `id` 的组合.
///
/// # 共享
///
/// 实现了 `Clone` —— 内部是 `Arc<wgpu::PipelineLayout>`, 克隆只增加引用
/// 计数. 多个调用点共享同一个布局时应当 clone handle, 而不是重新调用
/// [`Self::new`] —— 后者会分配新 `id`, 使管线缓存生成重复条目.
///
/// # 相等性
///
/// **不实现 `PartialEq`**. 两个 handle 是否「相等」在语义上有歧义:
/// 同一个布局的两份 clone 应视为相等 (`id` 相同), 不同布局即使底层 wgpu
/// 对象相同也不该视为相等. 需要比较时改用 [`Self::id`].
pub struct LayoutHandle {
	id: u64,
	layout: Arc<wgpu::PipelineLayout>,
}

impl LayoutHandle {
	/// 创建管线布局并登记新 `id`.
	///
	/// # 行为
	///
	/// - 分配一个新的 `id`.
	/// - 调用 [`wgpu::Device::create_pipeline_layout`] 建出布局对象.
	///
	/// # 参数
	///
	/// - **`device`**: 用于创建布局的设备.
	/// - **`groups`**: `bind group layout` 列表. 下标即 group 编号 ——
	///   `groups[0]` 对应 group 0, `groups[1]` 对应 group 1, 依此类推.
	///
	/// # 空 group
	///
	/// wgpu 30 起 `bind_group_layouts` 的元素类型变为
	/// `Option<&BindGroupLayout>` —— 传入 `None` 表示该 group 编号空缺.
	/// 本函数把传入的 `&[&BindGroupLayout]` 全部包装成 `Some`, 因此不支持
	/// 跳过某个 group 编号.
	///
	/// 需要跳号 (例如只用 group 0 和 group 2, 空出 group 1) 时, 直接调用
	/// wgpu 原生 API.
	///
	/// # 重复创建
	///
	/// 每次调用都会生成**新** handle (新 `id`, 新 wgpu 对象). 需要复用时保存
	/// 返回的 handle 并 clone, **不要**重新调用本函数.
	pub fn new(device: &wgpu::Device, groups: &[&wgpu::BindGroupLayout]) -> Self {
		let id = NEXT_LAYOUT_ID.fetch_add(1, Ordering::Relaxed);

		// wgpu 30 起需要包装为 `Option`.
		let groups_owned: Vec<Option<&wgpu::BindGroupLayout>> =
			groups.iter().map(|g| Some(*g)).collect();

		let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			// `label` 暂不提供 —— 多数场景下用 `id` 已足以区分.
			// 若确实需要 `label`, 再给 `new` 增加一个参数.
			label: None,
			bind_group_layouts: &groups_owned,
			// wgpu 30 起把 `push_constant_ranges` 改名为 `immediate_size`.
			// 值 0 表示不使用 immediate 数据.
			immediate_size: 0,
		});

		Self {
			id,
			layout: Arc::new(layout),
		}
	}

	/// 布局的唯一 `id`.
	///
	/// 用作 [`crate::state::PipelineDesc`] 的 `key`.
	pub fn id(&self) -> u64 {
		self.id
	}

	/// 返回底层的 wgpu 管线布局.
	pub fn raw(&self) -> &wgpu::PipelineLayout {
		&self.layout
	}
}

impl Clone for LayoutHandle {
	fn clone(&self) -> Self {
		Self {
			id: self.id,
			layout: Arc::clone(&self.layout),
		}
	}
}

impl std::fmt::Debug for LayoutHandle {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		// 不打印底层 wgpu 对象 —— 那只是无意义的内部细节.
		f.debug_struct("LayoutHandle")
			.field("id", &self.id)
			.finish_non_exhaustive()
	}
}
