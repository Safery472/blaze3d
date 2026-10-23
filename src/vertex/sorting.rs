//! 四边形图元的排序策略.
//!
//! 半透明绘制必须按从远到近的顺序进行, 否则混合结果会因深度冲突而
//! 错乱. 本模块提供 [`VertexSorting`][] `trait` 以及两个常用实现:
//!
//! - [`DistanceSorting`][]: 按到某个定点的距离排序, 一般用于透视相机.
//! - [`OrthographicZSorting`][]: 按 Z 轴倒序排列, 用于正交相机.
//!
//! # 与 `builder.rs` 的配合
//!
//! 排序不会在 [`crate::vertex::builder::Building`][] 中自动发生, 而是
//! 由 [`crate::vertex::builder::Building::set_quad_sorting`][] 显式开启.
//! 设置之后, `Building::end` 会做两件事:
//!
//! 1. 对 `crate::vertex::builder::Building::make_quad_sorting_points`
//!    算出的中心点调用 [`VertexSorting::sort`][];
//! 2. 依据返回的索引生成显式索引数据, 并把 `DrawState::sequential_index`
//!    置为 false.
//!
//! 若从未设置排序, 索引由共享索引缓冲提供, 此时 `sequential_index`
//! 保持为 true.

/// 图元绘制顺序的排序策略.
///
/// 实现者接收一组图元中心点, 并返回图元的绘制顺序. 顺序约定为
/// **从远到近** —— 越远的图元排在越前面, 也就先被绘制.
///
/// # 为什么必须「从远到近」
///
/// 半透明混合 (`src_alpha * src + (1 - src_alpha) * dst`) 的结果取决于
/// 绘制顺序. 先画远的, 再画近的, 近处的透明像素就会覆盖远处的颜色,
/// 从而得到符合视觉的前后关系. 顺序反过来则会把远处叠在近处之上.
///
/// # 实现者注意
///
/// - 返回的 `Vec<u32>` 长度应当等于 `points.len()`.
/// - 每个元素是**图元序号**, 而非顶点序号; 顶点序号是
///   `crate::vertex::builder::generate_sorted_quad_indices` 依据图元
///   序号展开出来的.
/// - 排序必须稳定, 或至少具备确定性 —— 相同输入多次调用应返回相同
///   结果, 否则同一场景的两次渲染之间可能出现细微差异.
///
/// # 线程安全
///
/// 要求实现 `Send + Sync`. 排序可能在渲染线程之外的线程上执行, 因此
/// 实现内部不应持有非 `Send` 的状态.
pub trait VertexSorting: Send + Sync {
	/// 给出各图元的绘制顺序.
	///
	/// # 参数
	///
	/// - `points`: 各图元的中心点. 其长度等于图元数量 (例如四边形批次
	///   的 `顶点数 / 4`).
	///
	/// # 返回值
	///
	/// 一个与 `points` 等长的索引数组, 用来表示绘制顺序. `result[0]`
	/// 是最先绘制的那个图元的序号.
	///
	/// **不要**返回长度与 `points` 不一致的数组 —— 调用方会按该长度
	/// 生成索引数据, 长度不匹配会让索引指向越界的顶点.
	fn sort(&self, points: &[[f32; 3]]) -> Vec<u32>;
}

/// 计算点到 `origin` 的平方距离.
///
/// 这里用平方距离而不再开方 —— 排序只关心顺序, 平方不影响结果, 还能
/// 省掉一次 `sqrt` 调用. 只有确实需要「实际距离」时才应开方.
fn squared_distance(point: [f32; 3], origin: [f32; 3]) -> f32 {
	let dx = point[0] - origin[0];
	let dy = point[1] - origin[1];
	let dz = point[2] - origin[2];
	dx * dx + dy * dy + dz * dz
}

/// 按到某个定点的距离排序.
///
/// 一般用于透视相机: 定点取相机的世界坐标, 距离越远的图元越先绘制.
///
/// # 用途
///
/// ```ignore
/// let mut b = builder.begin(DrawMode::Quads, format);
/// b.set_quad_sorting(Box::new(DistanceSorting::new(camera_pos)));
/// // ... 写入四边形 ...
/// let rendered = b.end();
/// ```
pub struct DistanceSorting {
	/// 距离计算所用的参考点, 一般取相机位置.
	pub origin: [f32; 3],
}

impl DistanceSorting {
	/// 用给定的参考点构造实例.
	pub const fn new(origin: [f32; 3]) -> Self {
		Self { origin }
	}
}

impl VertexSorting for DistanceSorting {
	fn sort(&self, points: &[[f32; 3]]) -> Vec<u32> {
		let origin = self.origin;

		// 先构造 0..n 的序号数组, 随后按距离对它排序.
		let mut indices: Vec<u32> = (0..points.len() as u32).collect();

		// 比较器给出「远者优先」的顺序.
		//
		// **不要**把比较方向改成 `a_dist.partial_cmp(&b_dist)` —— 那会
		// 按从近到远排列, 与「先画远的」正好相反. 这里刻意采用
		// `b_dist.partial_cmp(&a_dist)` 的方向.
		//
		// `partial_cmp` 返回 `None` (NaN) 时用 `Equal` 兜底. 输入点含
		// NaN 时排序结果不再确定, 但不会 panic. 若上游可能产生 NaN,
		// 应在写入顶点数据之前就处理掉.
		indices.sort_by(|&a, &b| {
			let da = squared_distance(points[a as usize], origin);
			let db = squared_distance(points[b as usize], origin);
			db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
		});

		indices
	}
}

/// 按 Z 轴的反向排序.
///
/// 一般用于正交相机: 相机朝向 -Z, 越远的图元 Z 值越小, 取负之后按
/// 降序排列, 得到的正是「远者优先」.
///
/// # 用途
///
/// 适用于 2D 界面 / 图标叠加一类场景: 每个图元的 Z 值本身就决定前后
/// 关系, 不需要相机位置.
pub struct OrthographicZSorting;

impl VertexSorting for OrthographicZSorting {
	fn sort(&self, points: &[[f32; 3]]) -> Vec<u32> {
		let mut indices: Vec<u32> = (0..points.len() as u32).collect();

		// 取 -z 之后再降序: z 越小 (即越远), -z 就越大, 排得也越前.
		//
		// **不要**把 `-points[...][2]` 改成 `points[...][2]` —— 那样
		// 会把顺序颠倒, 半透明叠加的结果随之出错.
		indices.sort_by(|&a, &b| {
			let za = -points[a as usize][2];
			let zb = -points[b as usize][2];
			zb.partial_cmp(&za).unwrap_or(std::cmp::Ordering::Equal)
		});

		indices
	}
}
