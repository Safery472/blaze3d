//! 长驻 GPU 的网格缓冲.
//!
//! [`GpuMesh`] 是把 [`RenderedBuffer`] 上传到 wgpu 缓冲之后得到的产物.
//! 它与 [`RenderedBuffer`] 的区别在于:
//!
//! - `RenderedBuffer` 是 CPU 端的字节数据, 用完即弃.
//! - [`GpuMesh`] 是 GPU 端的缓冲, 可以长期持有, 反复绘制, 并支持
//!   原地更新.
//!
//! # 使用场景
//!
//! - **静态几何** —— 地形块, 建筑, 长驻模型: 上传一次, 反复绘制.
//! - **可更新几何** —— 需要每帧或每隔几帧修改顶点内容, 但布局保持
//!   不变 (例如区块重建, 粒子位置更新).
//!
//! 只画一次的数据 (例如每帧变化一次的 HUD 元素) 不需要 [`GpuMesh`] ——
//! 先用 [`RenderedBuffer::upload`] 上传, 然后直接绘制即可.
//!
//! # 容量
//!
//! [`GpuMesh`] 的顶点数不得超过
//! [`crate::system::SharedIndexBuffers`] 的 `max_vertices` —— 否则用共享
//! 索引绘制时会读到越界数据. 越界不会 panic, 但绘制结果会出错.
//!
//! [`RenderedBuffer`]: RenderedBuffer
//! [`RenderedBuffer::upload`]: RenderedBuffer::upload

use std::sync::Arc;

use crate::system::error::{Error, Result};
use crate::vertex::{DrawMode, RenderedBuffer, VertexFormat};

/// GPU 端的一个网格.
///
/// 它持有顶点缓冲与可选的索引缓冲, 并记录绘制所需的元信息.
///
/// # 索引策略
///
/// [`Self::index_buffer`] 为 `None` 表示自身不持有索引 —— 此时绘制应
/// 使用 [`crate::system::SharedIndexBuffers`] 中按 `mode` 取出的共享
/// 索引. 为 `Some` 时使用自带索引 (例如 `Quads` 排序后的显式索引).
///
/// # 生命周期
///
/// 缓冲随 `GpuMesh` 一同 drop, 无需显式释放.
pub struct GpuMesh {
	vertex_buffer: wgpu::Buffer,

	/// 索引缓冲; 为 `None` 时改用共享索引.
	index_buffer: Option<wgpu::Buffer>,

	mode: DrawMode,
	vertex_count: u32,
	index_count: u32,
	format: Arc<VertexFormat>,
}

impl GpuMesh {
	/// 由 [`RenderedBuffer`] 构造网格.
	///
	/// # 行为
	///
	/// - 顶点数据非空时, 建立 vertex buffer 并上传.
	/// - 索引数据非空时, 建立 index buffer 并上传.
	/// - 记录格式, 模式, 顶点数与索引数.
	///
	/// # 错误
	///
	/// 顶点数为 0 时返回 [`Error::EmptyMesh`] —— 空网格无法绘制, 与其
	/// 创建一个长度为 0 的缓冲 (wgpu 会拒绝), 不如在这里直接拒绝.
	///
	/// # 缓冲大小
	///
	/// wgpu 要求 `Buffer` 的 `size` 是 4 的倍数. 顶点数据通常已经对齐
	/// (`stride` 通常是 4 的倍数), 但 `RenderedBuffer` 并不保证这一点,
	/// 因此上传时向上取整到 4.
	pub fn new(
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		buffer: &RenderedBuffer,
	) -> Result<Self> {
		let state = buffer.draw_state();

		if state.vertex_count == 0 {
			return Err(Error::EmptyMesh);
		}

		let vertex_buffer = create_buffer(
			device,
			queue,
			"gpu_mesh_vertices",
			buffer.vertex_data(),
			wgpu::BufferUsages::VERTEX,
		);

		let index_buffer = if buffer.index_data().is_empty() {
			None
		} else {
			Some(create_buffer(
				device,
				queue,
				"gpu_mesh_indices",
				buffer.index_data(),
				wgpu::BufferUsages::INDEX,
			))
		};

		Ok(Self {
			vertex_buffer,
			index_buffer,
			mode: state.mode,
			vertex_count: state.vertex_count,
			index_count: state.index_count,
			format: state.format.clone(),
		})
	}

	/// 就地更新顶点数据.
	///
	/// # 前置条件
	///
	/// 新数据的布局必须与原始数据**完全一致**, 具体包括:
	///
	/// - 与原始数据相同的 [`VertexFormat`].
	/// - 与原始数据相同的 [`DrawMode`].
	/// - 相同的顶点数量.
	/// - 相同的索引数量.
	/// - **索引的有无也须一致**: 原始数据带索引缓冲时新数据也必须带,
	///   反之亦然 (即 `sequential_index` 相同).
	///
	/// 可以变化的只有顶点内容与索引内容; 任一条件不满足, 本方法即
	/// 返回 [`Error::MeshLayoutMismatch`].
	///
	/// # 用途
	///
	/// 区块重建: 顶点位置发生变化, 但拓扑 (四边形数量, 顶点数, 索引数)
	/// 没变. 原地更新可以避免重新分配缓冲, 也避免重建 [`GpuMesh`].
	///
	/// # 为什么必须校验索引存在性
	///
	/// `index_count` 相同并不代表索引来源相同 —— `Quads` 启用排序与
	/// 不启用排序时索引数都是 `quad_count * 6`, 但前者走自带索引, 后者
	/// 走共享索引. 只比较 `index_count` 会让这种组合通过校验: 若原始
	/// 数据是排序索引 (自带), 新数据未排序 (共享), 写入时会跳过索引
	/// 覆盖, 于是绘制时用**旧索引**去读**新顶点**, 拓扑完全错乱; 反之
	/// 若原始是共享索引, 新数据想用自带索引, 又会因为 `index_buffer`
	/// 为 `None` 而无法写入, 渲染仍然走共享索引 —— 而调用方误以为
	/// 已经切换到排序后的顺序.
	///
	/// 因此这里必须显式比较 `index_buffer.is_some()` 与
	/// `!index_data.is_empty()` 是否一致.
	///
	/// # 与 `new` 的关系
	///
	/// 布局不匹配时, 调用方应改用 [`Self::new`] 重建, 而不是强行更新
	/// —— 后者会破坏 `GpuMesh` 的内部一致性.
	pub fn update(&mut self, queue: &wgpu::Queue, buffer: &RenderedBuffer) -> Result<()> {
		let state = buffer.draw_state();

		if state.format != self.format {
			return Err(Error::MeshLayoutMismatch("format"));
		}
		if state.mode != self.mode {
			return Err(Error::MeshLayoutMismatch("mode"));
		}
		if state.vertex_count != self.vertex_count {
			return Err(Error::MeshLayoutMismatch("vertex_count"));
		}
		if state.index_count != self.index_count {
			return Err(Error::MeshLayoutMismatch("index_count"));
		}

		// 索引来源必须一致 —— 理由见方法文档的「为什么必须校验索引
		// 存在性」一节. 让 `is_some()` 与 `!is_empty()` 对齐是 `GpuMesh`
		// 的内部不变量, 见 `new` 中 `index_buffer` 的构造分支.
		let new_has_index = !buffer.index_data().is_empty();
		if self.index_buffer.is_some() != new_has_index {
			return Err(Error::MeshLayoutMismatch("index_buffer presence"));
		}

		// 顶点数据: 直接整段覆盖.
		//
		// `write_buffer` 的 `size` 由调用方保证不超过缓冲大小 —— 顶点
		// 数已经校验相同, 因此字节数也相同.
		queue.write_buffer(&self.vertex_buffer, 0, buffer.vertex_data());

		// 索引数据: 只有两边都存在时才有内容可写. 前面的存在性校验
		// 已经保证 `Some` 与 `!is_empty` 一一对应.
		if let Some(index_buffer) = &self.index_buffer {
			queue.write_buffer(index_buffer, 0, buffer.index_data());
		}

		Ok(())
	}

	// ================ 字段访问 ================

	/// 顶点缓冲的引用.
	pub fn vertex_buffer(&self) -> &wgpu::Buffer {
		&self.vertex_buffer
	}

	/// 索引缓冲; 为 `None` 时使用共享索引.
	pub fn index_buffer(&self) -> Option<&wgpu::Buffer> {
		self.index_buffer.as_ref()
	}

	/// 网格的图元模式.
	pub fn mode(&self) -> DrawMode {
		self.mode
	}

	/// 网格的顶点数.
	pub fn vertex_count(&self) -> u32 {
		self.vertex_count
	}

	/// 网格的索引数.
	pub fn index_count(&self) -> u32 {
		self.index_count
	}

	/// 网格所用的顶点格式.
	pub fn format(&self) -> &VertexFormat {
		&self.format
	}
}

/// 可在多个线程间共享的网格句柄.
///
/// 多个绘制列表可以共享同一个网格 —— 例如同一棵树在视野中出现多次.
/// 用 `Arc` 包装即可避免重复上传.
pub type SharedGpuMesh = Arc<GpuMesh>;

/// 由字节数据创建 wgpu 缓冲.
///
/// # 行为
///
/// - 长度向上取整到 4 的倍数 (wgpu 的要求).
/// - 创建用途为 `usage | COPY_DST` 的缓冲.
/// - 把数据写入缓冲.
///
/// # 参数
///
/// - `usage` —— 目标用途 (`VERTEX` 或 `INDEX`), 内部会与 `COPY_DST`
///   一起使用.
fn create_buffer(
	device: &wgpu::Device,
	queue: &wgpu::Queue,
	label: &str,
	data: &[u8],
	usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
	let size = ((data.len() as u64) + 3) & !3;

	let buffer = device.create_buffer(&wgpu::BufferDescriptor {
		label: Some(label),
		size,
		usage: usage | wgpu::BufferUsages::COPY_DST,
		mapped_at_creation: false,
	});

	queue.write_buffer(&buffer, 0, data);
	buffer
}
