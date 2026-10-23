//! 共享的顺序索引缓冲.
//!
//! [`DrawMode`] 中的 `Quads` / `Triangles` / `Lines` 三种模式使用**顺序
//! 索引**: 索引值与顶点序号之间存在固定关系, 无需为每个网格单独存储.
//! 本模块预先生成这三种模式的索引缓冲, 供全部网格共享.
//!
//! # 为什么共享
//!
//! 一个 `Quads` 网格的 4 个顶点对应 6 个索引 (`[0,1,2,2,3,0]`). 若
//! 每个网格都保存自己的索引, 索引数据量便会与网格数量成正比; 共享
//! 缓冲则让索引量只与「最大顶点数」成正比, 与实际网格数无关.
//!
//! # 与显式索引的区别
//!
//! [`crate::vertex::DrawState::sequential_index`] 为 `true` 时使用本模块
//! 的共享索引; 为 `false` 时使用 [`crate::vertex::RenderedBuffer`] 自带
//! 的显式索引 (例如 `Quads` 排序后的结果).
//!
//! # 索引格式
//!
//! 所有缓冲都使用 `Uint32` 索引. 需要 `Uint16` 时应改造本模块 —— 当前
//! 不提供这一选项, 因为顶点数上限由 `max_vertices` 决定, 通常远大于
//! `u16::MAX` (即 65535).

use crate::vertex::DrawMode;

/// 三种绘制模式共用的索引缓冲.
///
/// # 容量
///
/// [`Self::new`] 接收 `max_vertices`, 生成的索引覆盖 `0..max_vertices`
/// 范围内的顶点. 网格的顶点数一旦超过 `max_vertices`, 绘制时就会读到
/// 越界索引 —— 调用方必须保证 `max_vertices` 是所有网格顶点数的上限.
///
/// 越界不会 panic, 但会画出错误内容. 这是 **有意设计**: 上限在
/// [`crate::system::RenderSystemDescriptor`] 中声明, 遵守它是调用方的
/// 责任.
pub struct SharedIndexBuffers {
	// `Quads` 模式的索引序列: [0,1,2,2,3,0, 4,5,6,6,7,4, ...]
	quads: wgpu::Buffer,
	quads_index_count: u32,

	// `Triangles` 模式的索引序列: [0,1,2, 3,4,5, ...]
	triangles: wgpu::Buffer,
	triangles_index_count: u32,

	// `Lines` 模式的索引序列: [0,1, 2,3, ...]
	lines: wgpu::Buffer,
	lines_index_count: u32,
}

impl SharedIndexBuffers {
	/// 按 `max_vertices` 创建共享索引.
	///
	/// # 行为
	///
	/// 共生成三个索引缓冲:
	///
	/// - `Quads` —— `max_vertices / 4 * 6` 个索引, 每 4 个顶点展开为 6
	///   个索引 `[b, b+1, b+2, b+2, b+3, b]`.
	/// - `Triangles` —— `max_vertices` 个索引, 顺序为 `0..N`.
	/// - `Lines` —— `max_vertices / 2 * 2` 个索引, 顺序为 `0..N`.
	///
	/// 除不尽的尾部顶点会被丢弃 —— 它们凑不成完整图元.
	///
	/// # 参数
	///
	/// - `device` —— 用于创建缓冲.
	/// - `max_vertices` —— 单个网格允许的顶点数上限.
	pub fn new(device: &wgpu::Device, max_vertices: u32) -> Self {
		let (quads, quads_index_count) = create_index_buffer(
			device,
			"shared_quad_indices",
			(0..max_vertices / 4).flat_map(|i| {
				let base = i * 4;
				[base, base + 1, base + 2, base + 2, base + 3, base]
			}),
		);

		// `Triangles` 的索引数与顶点数相同 —— `DrawMode::index_count`
		// 对 `Triangles` 直接返回 `vertex_count`, **不**做
		// `vertex_count / 3 * 3` 取整. 所以共享缓冲必须覆盖到
		// `max_vertices`, 不能只生成 `max_vertices / 3 * 3`.
		//
		// **不要**改成 `max_vertices / 3 * 3`: 当 `max_vertices` 不是 3
		// 的倍数, 且某网格的 `vertex_count == max_vertices` 时,
		// `draw_indexed(0..index_count)` 会越过索引缓冲末尾, 触发 wgpu 的
		// 隐式边界检查失败 (表现为绘制丢失或后端报错).
		let (triangles, triangles_index_count) = create_index_buffer(
			device,
			"shared_triangle_indices",
			0..max_vertices,
		);

		let (lines, lines_index_count) = create_index_buffer(
			device,
			"shared_line_indices",
			0..max_vertices / 2 * 2,
		);

		Self {
			quads,
			quads_index_count,
			triangles,
			triangles_index_count,
			lines,
			lines_index_count,
		}
	}

	/// 按绘制模式取出对应的索引缓冲.
	///
	/// # 返回值
	///
	/// - `Some((buffer, index_count))` —— 该模式存在共享索引.
	/// - `None` —— 该模式走**顶点直接绘制**, `TriangleStrip` 与
	///   `LineStrip` 不使用索引.
	///
	/// # 使用
	///
	/// ```ignore
	/// if let Some((buffer, count)) = shared.buffer_for(mode) {
	///     pass.set_index_buffer(buffer.slice(..), wgpu::IndexFormat::Uint32);
	///     pass.draw_indexed(0..count, 0, 0..1);
	/// } else {
	///     pass.draw(0..vertex_count, 0..1);
	/// }
	/// ```
	pub fn buffer_for(&self, mode: DrawMode) -> Option<(&wgpu::Buffer, u32)> {
		match mode {
			DrawMode::Quads => Some((&self.quads, self.quads_index_count)),
			DrawMode::Triangles => Some((&self.triangles, self.triangles_index_count)),
			DrawMode::Lines => Some((&self.lines, self.lines_index_count)),
			// 条带模式没有顺序索引 —— 顶点顺序本身就是绘制顺序.
			DrawMode::TriangleStrip | DrawMode::LineStrip => None,
		}
	}
}

/// 由索引迭代器创建 wgpu 缓冲.
///
/// # 行为
///
/// - 把索引收集进 `Vec<u32>`.
/// - 创建用途为 `INDEX | COPY_DST` 的缓冲.
/// - 按小端字节序写入索引.
///
/// # 空索引的处理
///
/// 当 `max_vertices` 小到没有任何完整图元时, `indices` 为空. 此时仍然
/// 创建一个最小长度 (4 字节) 的缓冲 —— wgpu 不允许长度为 0 的缓冲.
/// 返回的 `count` 为 0, 调用方绘制 `0..0` 即等同于无操作.
///
/// **不要**因此提前返回 `None` —— 保留空缓冲能让 `buffer_for` 的签名
/// 更简单 (始终为 `Some`), 空绘制的责任留给调用方.
fn create_index_buffer(
	device: &wgpu::Device,
	label: &str,
	indices: impl IntoIterator<Item = u32>,
) -> (wgpu::Buffer, u32) {
	let indices: Vec<u32> = indices.into_iter().collect();
	let count = indices.len() as u32;

	// wgpu 不接受 `size` 为 0 的缓冲, 因此空索引时创建最小长度缓冲.
	let byte_len = (indices.len() * size_of::<u32>()).max(4) as u64;

	let buffer = device.create_buffer(&wgpu::BufferDescriptor {
		label: Some(label),
		size: byte_len,
		usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
		// 直接映射写入 —— 比 `queue.write_buffer` 少一次拷贝. 这里只在
		// 初始化阶段用一次, 不存在并发问题.
		mapped_at_creation: true,
	});

	// 空索引时跳过写入 —— 映射区段长度为 4, 其内容无关紧要.
	if !indices.is_empty() {
		// 从 wgpu 30 开始, `get_mapped_range_mut` 返回 `Result`.
		let mut view = buffer
			.slice(..)
			.get_mapped_range_mut()
			.expect("failed to map index buffer for writing");
		// `bytemuck::cast_slice` 把 `&[u32]` 转成 `&[u8]`, 字节序取决于
		// 平台 —— 主流平台都是小端, 因此无需显式转换.
		view.copy_from_slice(bytemuck::cast_slice(&indices));
	}
	buffer.unmap();

	(buffer, count)
}
