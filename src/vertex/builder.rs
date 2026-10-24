// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 顶点缓冲构建器, 负责拼装顶点字节.
//!
//! 从零拼装顶点数据的完整流程:
//!
//! 1. 调用 [`BufferBuilder::begin`][] 借出构建状态, 得到 [`Building`][].
//! 2. 借助 [`VertexConsumer`][] 的方法逐个写入顶点.
//! 3. 调用 [`Building::end`][] 消耗构建状态, 产出 [`RenderedBuffer`][].
//! 4. [`RenderedBuffer::upload`][] 把数据上传到 `wgpu` 缓冲.
//!
//! # `Typestate` 设计
//!
//! 构建状态由 [`Building`][] 独占表达, 它借用 [`BufferBuilder`][].
//! 旧版「重复 begin」「未开始就写入」这类状态机错误在编译期就被
//! 排除: 拿不到 `Building` 就无法调用 `vertex`, 而 `BufferBuilder`
//! 被借用期间也无法再次 `begin`.
//!
//! # 内存布局
//!
//! 所有顶点数据连续存放在 `BufferBuilder::bytes` 中. [`Building`][]
//! 用 `next_element_byte` 追踪当前写入位置, 用 `element_index` 追踪
//! 当前元素在格式中的位置. 两者共同保证写入顺序与格式声明严格
//! 一致.
//!
//! # 索引策略
//!
//! 大多数模式 (`Quads` / `Triangles` / `Lines`) 采用**共享顺序索引**,
//! 数据里不保存索引. 只有 `Quads` 在启用排序时才生成显式索引数据,
//! 此时 `DrawState::sequential_index == false`.

use std::sync::Arc;

use super::consumer::VertexConsumer;
use super::element::{ElementType, ElementUsage, VertexFormatElement};
use super::error::{Error, Result};
use super::format::VertexFormat;
use super::sorting::VertexSorting;

/// 图元的组装模式.
///
/// **不含 `TriangleFan`**: `wgpu` 没有对应拓扑, 与其在运行时
/// 降级为 `TriangleList`, 不如干脆不提供.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrawMode {
	/// 独立线段, 每 2 个顶点构成一条.
	Lines,
	/// 连续的折线, 相邻顶点共享.
	LineStrip,
	/// 独立的三角形, 每 3 个顶点一个.
	Triangles,
	/// 连续的三角带.
	TriangleStrip,
	/// 独立的四边形, 每 4 个顶点一个, 内部展开为 6 个索引.
	Quads,
}

impl DrawMode {
	/// 由顶点数换算索引数.
	///
	/// # 注意
	///
	/// 这里的计数针对**共享顺序索引**. `Quads` 与 `Lines` 会改变
	/// 索引数 (4 顶点展开为 6 索引, 2 顶点仍为 2 索引), 其余模式的
	/// 索引数与顶点数相同.
	///
	/// 上层若改用显式索引, 索引数由 `RenderedBuffer::index_data`
	/// 决定, **不**用这个函数.
	pub const fn index_count(self, vertex_count: u32) -> u32 {
		match self {
			// 每 2 顶点一条线段 —— 索引数即偶数化后的顶点数.
			//
			// **不要**把它与 Quads 合并: 两者的展开规则不同. Quads 的
			// 4 顶点 → 6 索引是四边形特有的; Lines 的 2 顶点 → 2 索引
			// 恰好等于顶点数. 合并会让 Lines 的索引数偏大, 绘制时读到
			// 越界索引.
			Self::Lines => vertex_count / 2 * 2,
			Self::Quads => vertex_count / 4 * 6,
			Self::LineStrip | Self::Triangles | Self::TriangleStrip => vertex_count,
		}
	}

	/// 单个图元包含的顶点数.
	///
	/// 与 `DrawMode` 的语义严格对应. **不要**用它推算索引数 ——
	/// 见 [`Self::index_count`][].
	pub const fn primitive_stride(self) -> u32 {
		match self {
			Self::Lines => 2,
			Self::LineStrip | Self::TriangleStrip => 1,
			Self::Triangles => 3,
			Self::Quads => 4,
		}
	}

	/// 映射到 `wgpu` 的图元拓扑.
	///
	/// `Quads` 映射为 `TriangleList`: 索引层面已展开为三角形, 拓扑
	/// 只需声明为三角形列表.
	pub const fn wgpu_topology(self) -> wgpu::PrimitiveTopology {
		match self {
			Self::Lines => wgpu::PrimitiveTopology::LineList,
			Self::LineStrip => wgpu::PrimitiveTopology::LineStrip,
			Self::Triangles | Self::Quads => wgpu::PrimitiveTopology::TriangleList,
			Self::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
		}
	}
}

/// 索引的标量类型.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexType {
	/// 16 位无符号索引. 索引数不超过 65535 时优先选用.
	Short,
	/// 32 位无符号索引.
	Int,
}

impl IndexType {
	/// 单个索引占用的字节数.
	///
	/// `Short` 为 2, `Int` 为 4.
	pub const fn bytes(self) -> u32 {
		match self {
			Self::Short => 2,
			Self::Int => 4,
		}
	}

	/// 按索引数选取能容纳它的最小索引类型.
	///
	/// 判据是 `index_count > u16::MAX`, 用位运算表达来避开比较
	/// 分支, 便于 `const` 求值.
	pub const fn least(index_count: u32) -> Self {
		if (index_count & 0xFFFF_0000) != 0 {
			Self::Int
		} else {
			Self::Short
		}
	}

	/// 转换为 `wgpu` 的索引格式.
	///
	/// `Short` 对应 `Uint16`, `Int` 对应 `Uint32`.
	pub const fn wgpu(self) -> wgpu::IndexFormat {
		match self {
			Self::Short => wgpu::IndexFormat::Uint16,
			Self::Int => wgpu::IndexFormat::Uint32,
		}
	}
}

/// 一次绘制所需的元信息.
///
/// 与 [`RenderedBuffer`][] 一同产出, 渲染层用它决定:
///
/// - 管线选择 (依据 `format`);
/// - 绘制拓扑 (依据 `mode`);
/// - 绘制范围 (依据 `index_count` / `index_type`);
/// - 是否使用共享索引 (依据 `sequential_index`).
#[derive(Debug, Clone)]
pub struct DrawState {
	/// 顶点格式. 与 [`BufferBuilder::begin`][] 收到的格式相同.
	pub format: Arc<VertexFormat>,

	/// 本批次的顶点数量.
	pub vertex_count: u32,

	/// 索引的总数量.
	///
	/// 使用共享顺序索引 (`sequential_index == true`) 时等于
	/// `mode.index_count(vertex_count)`; 否则等于实际生成的索引数.
	pub index_count: u32,

	/// 本批次使用的图元模式.
	pub mode: DrawMode,

	/// 索引数据使用的标量类型.
	pub index_type: IndexType,

	/// 是否使用共享的顺序索引.
	///
	/// - `true`: `RenderedBuffer::index_data` 为空, 索引由渲染层的
	///   共享顺序索引缓冲按 `mode` 提供.
	/// - `false`: 必须使用 `RenderedBuffer::index_data` 里的索引.
	pub sequential_index: bool,
}

impl DrawState {
	/// 顶点数据占用的字节数.
	///
	/// 等于 `vertex_count * format.vertex_size()`.
	pub fn vertex_buffer_size(&self) -> u32 {
		self.vertex_count * self.format.vertex_size()
	}

	/// 索引数据的总字节数.
	///
	/// 使用共享顺序索引 (`sequential_index == true`) 时返回 0 ——
	/// 此时没有索引数据需要上传.
	pub fn index_buffer_size(&self) -> u32 {
		if self.sequential_index {
			0
		} else {
			self.index_count * self.index_type.bytes()
		}
	}
}

/// 一次构建的产出.
///
/// 由 [`Building::end`][] 产出. 数据以裸字节形式留在内存中,
/// 何时上传到 `GPU` 由调用方决定.
pub struct RenderedBuffer {
	draw_state: DrawState,

	// 顶点字节数据.
	//
	// **不保证 4 字节对齐**: BufferBuilder 内部是 `Vec<u8>`, 按字节
	// 直接追加. 上传到 `wgpu` 缓冲时由 `upload` 补齐对齐.
	vertex_data: Vec<u8>,

	// 索引字节数据.
	//
	// 只有 Quads 启用排序时才非空, 且此时
	// `draw_state.sequential_index` 为 false. 其余情形为空, 由共享
	// 索引缓冲兜底.
	index_data: Vec<u8>,
}

impl RenderedBuffer {
	/// 本次构建的绘制元信息.
	pub fn draw_state(&self) -> &DrawState {
		&self.draw_state
	}

	/// 顶点字节数据.
	///
	/// 长度恰好是 `draw_state.vertex_buffer_size()`.
	pub fn vertex_data(&self) -> &[u8] {
		&self.vertex_data
	}

	/// 索引字节数据.
	///
	/// `draw_state.sequential_index == true` 时为空, 此时应改用
	/// 共享索引.
	pub fn index_data(&self) -> &[u8] {
		&self.index_data
	}

	/// 是否没有任何顶点.
	///
	/// 等价于 `draw_state.vertex_count == 0`.
	pub fn is_empty(&self) -> bool {
		self.draw_state.vertex_count == 0
	}

	/// 上传到 `wgpu` 缓冲.
	///
	/// # 行为
	///
	/// - 顶点数据非空: 创建 vertex buffer 并写入.
	/// - 索引数据非空: 创建 index buffer 并写入.
	/// - 对应数据为空时该字段为 `None`.
	///
	/// # 关于 size 对齐
	///
	/// `wgpu::Buffer` 的 size 必须是 4 的倍数, 否则部分后端会在
	/// `create_buffer` 时直接 panic, 因此这里统一向上取整到 4.
	///
	/// **不要**把 size 改回原始长度.
	pub fn upload(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> UploadedBuffer {
		let vertex_buffer = if self.vertex_data.is_empty() {
			None
		} else {
			let size = align4(self.vertex_data.len() as u64);
			let buf = device.create_buffer(&wgpu::BufferDescriptor {
				label: Some("blaze3d_vertex_buffer"),
				size,
				usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
				mapped_at_creation: false,
			});
			queue.write_buffer(&buf, 0, &self.vertex_data);
			Some(buf)
		};

		let index_buffer = if self.index_data.is_empty() {
			None
		} else {
			let size = align4(self.index_data.len() as u64);
			let buf = device.create_buffer(&wgpu::BufferDescriptor {
				label: Some("blaze3d_index_buffer"),
				size,
				usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
				mapped_at_creation: false,
			});
			queue.write_buffer(&buf, 0, &self.index_data);
			Some(buf)
		};

		UploadedBuffer {
			vertex_buffer,
			index_buffer,
			draw_state: self.draw_state.clone(),
		}
	}
}

/// 把字节数向上取整到 4 的倍数.
fn align4(n: u64) -> u64 {
	(n + 3) & !3
}

/// 已上传到 `wgpu` 的缓冲.
///
/// 与 [`RenderedBuffer`][] 的差别是数据已写入 `GPU` 缓冲, 不再持有
/// `CPU` 端字节. `vertex_buffer` / `index_buffer` 的生命周期由
/// `wgpu` 管理.
pub struct UploadedBuffer {
	/// 顶点缓冲. 顶点数据为空时为 `None`.
	pub vertex_buffer: Option<wgpu::Buffer>,

	/// 索引缓冲. 索引数据为空时为 `None`.
	///
	/// `draw_state.sequential_index == true` 时同样为 `None`, 此时
	/// 渲染层须改用共享索引.
	pub index_buffer: Option<wgpu::Buffer>,

	/// 随缓冲一同上传的绘制元信息.
	pub draw_state: DrawState,
}

/// 构建器本身.
///
/// 只持有字节缓冲. 真正的构建状态在 [`Building`][] 里 —— `begin`
/// 时借用本结构, `end` / `discard` 时归还.
///
/// # 复用
///
/// 同一个 `BufferBuilder` 可以反复 `begin` / `end`, 但**容量不跨
/// 批次保留**: `end` 把数据移交给 [`RenderedBuffer`][] 后, 本结构
/// 的缓冲会重置为一个新的空 `Vec` (容量 0), 下次 `begin` 从头
/// 分配.
///
/// 这是有意的取舍 —— 让 `RenderedBuffer` 拥有独立数据, 不与
/// `BufferBuilder` 的生命周期耦合. 若要减少分配, 请复用
/// `RenderedBuffer` (上传后丢弃), 或改用长驻 `GPU` 的网格.
pub struct BufferBuilder {
	// 顶点数据与索引数据共用的缓冲.
	//
	// 布局为 [顶点数据][索引数据]; 索引数据仅在 Quads 排序时存在.
	// Building 以 next_element_byte 追踪顶点数据末尾, 索引数据追加
	// 在它后面.
	//
	// **不要**在 Building 存活期间从外部访问该字段 —— 会破坏
	// 「next_element_byte 与 len 的关系」这一不变量.
	bytes: Vec<u8>,
}

impl Default for BufferBuilder {
	fn default() -> Self {
		Self::new()
	}
}

impl BufferBuilder {
	/// 创建一个空的构建器.
	///
	/// 不预分配字节缓冲, 首次写入时按需增长.
	pub const fn new() -> Self {
		Self { bytes: Vec::new() }
	}

	/// 创建构建器, 并预分配 `bytes` 字节容量.
	///
	/// 预分配可以减少大批量写入时的重新分配. 注意容量不跨批次
	/// 保留 —— 见 [`BufferBuilder`][] 的「复用」一节.
	pub fn with_capacity(bytes: usize) -> Self {
		Self {
			bytes: Vec::with_capacity(bytes),
		}
	}

	/// 开始一次构建.
	///
	/// # 行为
	///
	/// 清空旧数据, 生成借用 `self` 的 [`Building`][]. 旧的
	/// [`Building`][] 存活期间无法再次调用 —— 借用检查器会拒绝.
	///
	/// # Panics
	///
	/// 传入的 `format` 若元素为空, 会在内部 `elements()[0]` 处
	/// panic. 但 [`VertexFormat::new`][] 已保证首元素存在且不是
	/// `Padding`, 因此**经公开 `API` 拿到的格式不会触发**.
	///
	/// [`VertexFormat::new`]: VertexFormat::new
	pub fn begin(&mut self, mode: DrawMode, format: VertexFormat) -> Building<'_> {
		// 清空复用容量. 若上次 end 已把数据移交出去, 这里只是把空 Vec
		// 的长度归零 —— 不涉及释放, 也不保留任何分配.
		self.bytes.clear();

		let format = Arc::new(format);

		// 取首个元素作为 current_element.
		//
		// 前置不变量:
		//   1. format 一定非空.
		//   2. 首元素一定不是 Padding.
		//
		// 两者都由 VertexFormat::new 保证. 若将来该不变量失效, 这行
		// 会 panic —— 这是有意的: 与其返回 Err 让调用方去处理一个
		// 不可能出现的错误, 不如直接暴露出来.
		let first = format.elements()[0];

		Building {
			owner: self,
			format,
			mode,
			current_element: first,
			element_index: 0,
			next_element_byte: 0,
			vertices: 0,
			default_color_set: false,
			default_rgba: [255; 4],
			sorting: None,
		}
	}
}

/// 正在进行的构建状态.
///
/// 通过 [`BufferBuilder::begin`][] 取得. **不能**从外部直接构造
/// —— 字段全部私有.
///
/// # 生命周期
///
/// 借用 [`BufferBuilder`][]. 调用 [`Self::end`][] 或
/// [`Self::discard`][] 时消耗自身, 归还借用.
///
/// # 写入协议
///
/// 所有 [`VertexConsumer`][] 方法内部都按同一顺序执行:
///
/// 1. 校验当前元素是否匹配方法的语义;
/// 2. 写入对应的字节;
/// 3. 经 `Self::advance` 推进到下一个元素.
///
/// 顺序错乱时在第 1 步返回 [`Error::WrongElement`][], 不写入
/// 任何字节.
pub struct Building<'a> {
	// 被借用的构建器本体. 所有字节都写进 owner.bytes.
	owner: &'a mut BufferBuilder,

	// 当前构建使用的格式. 用 Arc 是为了让 end 之后能直接把共享
	// 句柄放进 DrawState, 不必 clone 一份完整格式.
	format: Arc<VertexFormat>,

	// 本次构建的图元模式.
	mode: DrawMode,

	// 当前等待写入的元素.
	//
	// 与 element_index 的联动关系:
	//   current_element == format.elements()[element_index]
	//
	// 初值是 elements[0]; 每次 advance 之后同步更新.
	current_element: VertexFormatElement,

	// 当前元素在 format.elements 里的下标.
	//
	// 归零表示一个顶点写满 —— 但**不要**用它做 end_vertex 的
	// 判定, 原因见 end_vertex 的注释.
	element_index: usize,

	// 下一个待写入字节在 owner.bytes 中的偏移.
	//
	// 从 0 起算, 每写完一个元素按其 byte_size 递增.
	// 顶点写满时恰好等于 vertices * format.vertex_size().
	next_element_byte: usize,

	// 已经提交的顶点数.
	vertices: u32,

	// 默认颜色是否已启用.
	default_color_set: bool,

	// 默认颜色的 RGBA 分量. default_color_set 为 false 时无意义.
	default_rgba: [u8; 4],

	// 四边形排序所用的策略.
	//
	// 只有 mode == Quads 时由 set_quad_sorting 填充, 其他模式恒为
	// None. 中心点不在写入过程中缓存 —— 见 set_quad_sorting 与
	// end 的注释.
	sorting: Option<Box<dyn VertexSorting>>,
}

impl<'a> Building<'a> {
	/// 本次构建的图元模式.
	pub fn mode(&self) -> DrawMode {
		self.mode
	}

	/// 已经提交的顶点数.
	pub fn vertex_count(&self) -> u32 {
		self.vertices
	}

	/// 是否还没有提交任何顶点.
	pub fn is_empty(&self) -> bool {
		self.vertices == 0
	}

	/// 设置四边形排序策略.
	///
	/// # 行为
	///
	/// - 非 `Quads` 模式下调用时**静默忽略** —— 排序只对四边形
	///   有意义.
	/// - 中心点在 [`Self::end`][] 时按**最终**顶点数据计算一次,
	///   因此在写入前或写入后调用都正确, 唯一约束是必须在 `end`
	///   之前.
	///
	/// # 与格式的关系
	///
	/// 计算中心点假设格式前 12 字节是 `Position` (`Float` x3) —— 与
	/// 所有 `formats::*` 便利格式的约定一致. 自定义格式若把
	/// `Position` 放在别处, 排序结果会错.
	///
	/// **不要**放宽这个假设: 要在 `Building` 里核对 `Position` 的位置
	/// 就得保存更多状态, 收益远小于代价.
	pub fn set_quad_sorting(&mut self, sorting: Box<dyn VertexSorting>) {
		if self.mode != DrawMode::Quads {
			return;
		}
		self.sorting = Some(sorting);
	}

	/// 收束构建, 产出可上传的缓冲.
	///
	/// # 行为
	///
	/// - 消耗 `self`, 把对 [`BufferBuilder`][] 的借用归还.
	/// - `Quads` 模式且启用排序时, 按**最终**顶点数据算出四边形
	///   中心点, 据此生成显式索引数据.
	/// - 其他模式下 `index_data` 为空, `draw_state.sequential_index`
	///   为 true.
	///
	/// # 容量
	///
	/// **`BufferBuilder` 的容量不会被保留**. 数据移交
	/// [`RenderedBuffer`][] 后, `BufferBuilder::bytes` 是一个全新的
	/// 空 `Vec`, 容量为 0, 下次 `begin` 需要重新分配.
	///
	/// 这是有意的取舍 —— 让 `RenderedBuffer` 拥有独立数据, 不与
	/// `BufferBuilder` 的生命周期绑在一起.
	pub fn end(self) -> RenderedBuffer {
		let format = self.format.clone();
		let vertex_count = self.vertices;
		let index_count = self.mode.index_count(vertex_count);
		let index_type = IndexType::least(index_count);

		let mut index_data = Vec::new();
		let mut sequential_index = true;

		// 排序只在 end 时按最终顶点数据做一次.
		//
		// **不要**把中心点缓存进 set_quad_sorting —— 那会让
		// index_data.len() 与实际图元数脱节: 若 set 之后又追加
		// 四边形, 缓存中的中心点会少于实际图元数, 索引随之
		// 不足, 而 DrawState::index_count 仍按 vertex_count 算.
		// 渲染层按 index_count 发起 draw, 就会越界读索引缓冲.
		if self.mode == DrawMode::Quads
			&& let Some(sorting) = &self.sorting
		{
			let points = self.make_quad_sorting_points();
			if !points.is_empty() {
				let order = sorting.sort(&points);
				index_data = generate_sorted_quad_indices(&order, index_type);
				sequential_index = false;
			}
		}

		// 截断到实际写入的长度.
		self.owner.bytes.truncate(self.next_element_byte);

		// 把数据转交给 RenderedBuffer.
		//
		// mem::take 把 owner.bytes 换成一个新空 Vec (容量 0), 原有
		// 分配随数据一起移出. 这意味着 **BufferBuilder 不保留容量**.
		//
		// 不要改成 `clone() + clear()`: 那样每次 end 都要复制一遍
		// 数据, 分配次数却没有减少, 没有收益.
		let vertex_data = std::mem::take(&mut self.owner.bytes);

		RenderedBuffer {
			draw_state: DrawState {
				format,
				vertex_count,
				index_count,
				mode: self.mode,
				index_type,
				sequential_index,
			},
			vertex_data,
			index_data,
		}
	}

	/// 放弃当前构建.
	///
	/// 等价于「清空数据 + 归还借用」, 不产出 [`RenderedBuffer`][].
	pub fn discard(self) {
		self.owner.bytes.clear();
	}

	// ================ 内部辅助方法 ================

	/// 确保 `owner.bytes` 至少有 `next_element_byte + additional`
	/// 字节的容量.
	///
	/// 增长策略: 按 2 的幂次扩张, 下限 [`GROWTH_SIZE`][]. 用几何
	/// 增长避免高写入量下的频繁 realloc.
	///
	/// **不要**改成逐字节 push —— 那会拖慢批量写入. 当前实现一次
	/// resize 后按需 copy, 均摊成本 `O(1)`.
	fn reserve(&mut self, additional: usize) {
		let needed = self.next_element_byte + additional;
		if self.owner.bytes.len() < needed {
			let target = needed.next_power_of_two().max(GROWTH_SIZE);
			self.owner.bytes.resize(target, 0);
		}
	}

	/// 从 `next_element_byte + offset` 处写入字节.
	fn put_bytes(&mut self, offset: usize, data: &[u8]) {
		self.reserve(offset + data.len());
		let pos = self.next_element_byte + offset;
		self.owner.bytes[pos..pos + data.len()].copy_from_slice(data);
	}

	/// 校验当前元素是否与期望一致.
	///
	/// 匹配三元组 `(usage, elem_type, count)`. 不一致时返回
	/// [`Error::WrongElement`][], 并带上实际元素.
	fn expect_element(
		&self,
		usage: ElementUsage,
		elem_type: ElementType,
		count: u32,
	) -> Result<()> {
		let actual = self.current_element;
		if actual.usage() == usage
			&& actual.elem_type() == elem_type
			&& actual.count() == count
		{
			Ok(())
		} else {
			Err(Error::WrongElement {
				actual,
				expected_usage: usage,
				expected_type: elem_type,
				expected_count: count,
			})
		}
	}

	/// 前进到下一个元素.
	///
	/// # 行为
	///
	/// - 跳过 `Padding`: 不写数据, 但仍推进 `next_element_byte`
	///   跨过它的字节.
	/// - 下一个元素若是 `Color` 且已设置默认颜色, 自动填入默认值.
	/// - 循环到第一个既非 `Padding` 也无需自动填充的元素时返回.
	///
	/// # 与 `format.rs` 的关系
	///
	/// `element_index` 与 `format.elements` 的顺序同步. 它回到 0 时
	/// 表示一个顶点写满. **不要**在这里修改 `vertices` —— 顶点计数
	/// 由 [`Self::end_vertex`][] 负责.
	fn advance(&mut self) -> Result<()> {
		// 克隆 Arc 而非直接借用 self.format: 借用 self.format 会让整个
		// 循环持有对 self 的不可变借用, 与循环体里需要 &mut self 的
		// self.put_bytes 冲突. Arc 克隆只加引用计数, 没有堆分配.
		let format = Arc::clone(&self.format);
		let elements = format.elements();
		let len = elements.len();

		loop {
			self.element_index = (self.element_index + 1) % len;
			self.next_element_byte += self.current_element.byte_size() as usize;
			let next = elements[self.element_index];

			match next.usage() {
				// Padding 不写数据, 只跳过字节. 跳过前先把
				// current_element 换成它, 下一个循环才能正确
				// 累加它的 byte_size.
				ElementUsage::Padding => {
					self.current_element = next;
					continue;
				}

				// Color 且已设置默认颜色: 自动填充.
				//
				// **不变量**: next.count() == 4,
				// next.elem_type() == UByte. 由 Color 的构造路径
				// 保证 —— 见 element.rs 的 color_u8x4 /
				// color_f32x4. 若将来支持 Float 颜色, 这里要按
				// 类型分别填充.
				ElementUsage::Color if self.default_color_set => {
					let [r, g, b, a] = self.default_rgba;
					self.current_element = next;
					self.put_bytes(0, &[r, g, b, a]);
					continue;
				}

				_ => {
					self.current_element = next;
					return Ok(());
				}
			}
		}
	}

	/// 计算四边形中心点, 供排序使用.
	///
	/// # 假设
	///
	/// 格式前 12 字节是 `Position` (`Float` x3). 这与所有
	/// `formats::*` 便利格式的约定一致.
	///
	/// # 实现说明
	///
	/// 取每个四边形第 0 与第 2 个顶点的位置求平均. 对矩形而言这与
	/// 几何中心一致; 对非平面四边形只是近似值, 排序够用.
	fn make_quad_sorting_points(&self) -> Vec<[f32; 3]> {
		let stride = self.format.vertex_size() as usize;
		let float_count = self.format.integer_size() as usize;
		let quad_stride = float_count * 4;

		let total = self.next_element_byte;
		let vertex_bytes = self.vertices as usize * stride;
		if vertex_bytes == 0 || vertex_bytes > total {
			return Vec::new();
		}

		// 把整个顶点数据按 f32 读出.
		//
		// f32::from_le_bytes 必须与写入时的 to_le_bytes 配对,
		// 不能改成 from_ne_bytes, 否则跨平台会不一致.
		let mut floats = Vec::with_capacity(total / 4);
		let mut i = 0;
		while i + 4 <= total {
			let b = &self.owner.bytes[i..i + 4];
			floats.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
			i += 4;
		}

		let quad_count = (self.vertices / 4) as usize;
		let mut points = Vec::with_capacity(quad_count);
		for qi in 0..quad_count {
			let base = qi * quad_stride;
			let v2 = base + float_count * 2;
			if v2 + 3 > floats.len() {
				break;
			}
			points.push([
				(floats[base] + floats[v2]) * 0.5,
				(floats[base + 1] + floats[v2 + 1]) * 0.5,
				(floats[base + 2] + floats[v2 + 2]) * 0.5,
			]);
		}
		points
	}

	/// 写入 16 位 `UV` 数据.
	///
	/// `overlay_coords` (层号 1) 与 `uv2` (层号 2) 共用此函数.
	/// 两者格式的标量类型不同: overlay 是 `Short` (层号可以为负),
	/// uv2 是 `UShort` (打包光照坐标). 由 `elem_type` 参数区分.
	///
	/// # 位模式
	///
	/// `i16` 与 `u16` 对相同位模式调用 `to_le_bytes` 的结果一致,
	/// 因此内部统一用 `u16` 写入, 由调用方先把值转换到目标位宽.
	fn put_uv_16bit(
		&mut self,
		u: u16,
		v: u16,
		index: u32,
		elem_type: ElementType,
	) -> Result<()> {
		let actual = self.current_element;
		if actual.usage() != ElementUsage::Uv
			|| actual.uv_index() != Some(index)
			|| actual.elem_type() != elem_type
			|| actual.count() != 2
		{
			return Err(Error::WrongElement {
				actual,
				expected_usage: ElementUsage::Uv,
				expected_type: elem_type,
				expected_count: 2,
			});
		}
		self.put_bytes(0, &u.to_le_bytes());
		self.put_bytes(2, &v.to_le_bytes());
		self.advance()
	}
}

impl VertexConsumer for Building<'_> {
	fn vertex(&mut self, x: f32, y: f32, z: f32) -> Result<()> {
		self.expect_element(ElementUsage::Position, ElementType::Float, 3)?;
		self.put_bytes(0, &x.to_le_bytes());
		self.put_bytes(4, &y.to_le_bytes());
		self.put_bytes(8, &z.to_le_bytes());
		self.advance()
	}

	fn color(&mut self, r: u8, g: u8, b: u8, a: u8) -> Result<()> {
		// 先检查默认颜色 —— 已设置就报错, 而不是静默覆盖.
		//
		// 这个顺序很重要: 若先检查元素类型, 在「已设默认颜色 +
		// Color 元素也正确」的组合下会返回 WrongElement, 掩盖真正
		// 的原因. 先报 DefaultColorActive, 调用方才能看到根因.
		if self.default_color_set {
			return Err(Error::DefaultColorActive);
		}
		self.expect_element(ElementUsage::Color, ElementType::UByte, 4)?;
		self.put_bytes(0, &[r, g, b, a]);
		self.advance()
	}

	fn uv(&mut self, u: f32, v: f32) -> Result<()> {
		let actual = self.current_element;
		if actual.usage() != ElementUsage::Uv
			|| actual.uv_index() != Some(0)
			|| actual.elem_type() != ElementType::Float
			|| actual.count() != 2
		{
			return Err(Error::WrongElement {
				actual,
				expected_usage: ElementUsage::Uv,
				expected_type: ElementType::Float,
				expected_count: 2,
			});
		}
		self.put_bytes(0, &u.to_le_bytes());
		self.put_bytes(4, &v.to_le_bytes());
		self.advance()
	}

	fn overlay_coords(&mut self, u: i32, v: i32) -> Result<()> {
		// 层号可以为负 —— 语义即 i16. 先 as i16 再 as u16 保留位模式,
		// **不要**直接 as u16: 那会按 32 位截断负数, 结果不同.
		self.put_uv_16bit(
			(u as i16) as u16,
			(v as i16) as u16,
			1,
			ElementType::Short,
		)
	}

	fn uv2(&mut self, u: i32, v: i32) -> Result<()> {
		// 打包光照坐标, 两个分量各 16 位无符号.
		// **不要**改成 `u as i16 as u16`: 位模式虽然相同, 但语义错误,
		// 会让读者以为光照坐标可以是有符号的.
		self.put_uv_16bit(u as u16, v as u16, 2, ElementType::UShort)
	}

	fn normal(&mut self, x: f32, y: f32, z: f32) -> Result<()> {
		self.expect_element(ElementUsage::Normal, ElementType::Byte, 3)?;
		// 先归一化到 `[-1, 1]` 区间, 再按 `i8` 存储.
		self.put_bytes(
			0,
			&[normal_int_value(x), normal_int_value(y), normal_int_value(z)],
		);
		self.advance()
	}

	fn bone_index(&mut self, indices: [u8; 4]) -> Result<()> {
		self.expect_element(ElementUsage::BoneIndex, ElementType::UByte, 4)?;
		self.put_bytes(0, &indices);
		self.advance()
	}

	fn bone_weight(&mut self, weights: [u8; 4]) -> Result<()> {
		self.expect_element(ElementUsage::BoneWeight, ElementType::UByte, 4)?;
		self.put_bytes(0, &weights);
		self.advance()
	}

	fn end_vertex(&mut self) -> Result<()> {
		// u32 上限防御. 提前检查, 以免后续计算溢出.
		if self.vertices == u32::MAX {
			return Err(Error::VertexOverflow);
		}

		let stride = self.format.vertex_size() as usize;

		// 判定顶点是否写满: 写完一个顶点后, 游标应恰好等于
		// (vertices + 1) * stride.
		//
		// **不要**改用 `element_index != 0`: 单元素格式下
		// element_index 写完即归零 (advance 中 `(0 + 1) % 1 == 0`),
		// 无法区分「一个元素都没写」与「刚写完一个顶点」. 因此当前
		// 判定基于字节游标, 对任意元素数都成立.
		let expected = (self.vertices as usize + 1) * stride;
		if self.next_element_byte != expected {
			let total = self.format.vertex_size();
			// 当前顶点已写入的字节数 —— 自顶点起始位置算起.
			//
			// saturating_sub 只是防御: 不变量保证
			// next_element_byte >= vertex_start, 但即便上游改动
			// 破坏了它, 错误路径本身也不该 panic.
			let vertex_start = self.vertices as usize * stride;
			let written = self.next_element_byte.saturating_sub(vertex_start) as u32;
			return Err(Error::IncompleteVertex {
				total,
				remaining: total.saturating_sub(written),
			});
		}

		self.vertices += 1;

		// 预留下一个顶点的空间, 避免下次写入时再检查.
		// stride 至少为 1 (format 非空, 且元素 byte_size >= 1).
		self.reserve(stride);
		Ok(())
	}

	fn default_color(&mut self, r: u8, g: u8, b: u8, a: u8) {
		self.default_rgba = [r, g, b, a];
		self.default_color_set = true;
	}

	fn unset_default_color(&mut self) {
		self.default_color_set = false;
	}
}

/// 把浮点法线归一化到 `u8` 字节.
///
/// 先 clamp 到 `[-1, 1]`, 乘以 127, 截断为 `i8`, 再按位与 `0xFF`
/// 转成 `u8`.
///
/// **不要**改成先乘 128 —— 语义边界是 127, 改了会与已有数据
/// 不一致.
fn normal_int_value(f: f32) -> u8 {
	let c = f.clamp(-1.0, 1.0);
	((c * 127.0) as i32 & 0xFF) as u8
}

/// 生成排序之后的四边形索引.
///
/// # 参数
///
/// - `order`: 排序后的四边形序号, 取自 [`VertexSorting::sort`][].
/// - `index_type`: 输出索引的字节宽度 (`Short` / `Int`).
///
/// # 输出
///
/// 每个四边形展开为 6 个索引: `[b, b+1, b+2, b+2, b+3, b]`,
/// 与共享顺序索引缓冲中的排列一致.
fn generate_sorted_quad_indices(order: &[u32], index_type: IndexType) -> Vec<u8> {
	let mut out = Vec::with_capacity(order.len() * 6 * index_type.bytes() as usize);
	for &sorted in order {
		let base = sorted * 4;
		for idx in [base, base + 1, base + 2, base + 2, base + 3, base] {
			match index_type {
				IndexType::Short => out.extend_from_slice(&(idx as u16).to_le_bytes()),
				IndexType::Int => out.extend_from_slice(&idx.to_le_bytes()),
			}
		}
	}
	out
}

/// 缓冲增长时的步长下限.
///
/// 小批次不会触发; 大批次首次增长之后按需倍增.
const GROWTH_SIZE: usize = 2 * 1024 * 1024;
