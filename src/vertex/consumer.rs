//! 顶点数据的写入协议与多路广播装饰器.
//!
//! 本模块以 [`VertexConsumer`][] 作为写入顶点数据的统一接口. 实现者按
//! [`crate::vertex::format::VertexFormat`][] 中元素的声明顺序逐个调用写入
//! 方法, 每次调用推进到下一个元素; 一个顶点的全部元素写完后, 再调用
//! [`VertexConsumer::end_vertex`][] 提交该顶点.
//!
//! # 写入顺序约定
//!
//! 写入方法必须严格遵循格式的声明顺序, 以 `position_color` 格式为例:
//!
//! ```rust
//! use blaze3d::vertex::{formats, BufferBuilder, DrawMode, VertexConsumer};
//!
//! let mut builder = BufferBuilder::with_capacity(4096);
//! let mut b = builder.begin(DrawMode::Quads, formats::position_color());
//!
//! b.vertex(0.0, 0.0, 0.0)?;
//! b.color(255, 255, 255, 255)?;
//! b.end_vertex()?;
//! # Ok::<(), blaze3d::vertex::Error>(())
//! ```
//!
//! 顺序错乱会返回 [`crate::vertex::Error::WrongElement`][], 而不会 panic.
//!
//! # 默认颜色
//!
//! 设置 [`VertexConsumer::default_color`][] 之后, 格式中的 `Color` 元素
//! 由实现者自动填充, 调用方无需再调 `color`. 若两者同时使用, 返回
//! [`crate::vertex::Error::DefaultColorActive`][].
//!
//! # 骨骼
//!
//! [`VertexConsumer::bone_index`][] 与 [`VertexConsumer::bone_weight`][]
//! 分别写入 4 个骨骼索引与 4 个权重, 与格式中的 `BoneIndex` /
//! `BoneWeight` 元素一一对应.
//!
//! # 自定义实现
//!
//! 除 [`crate::vertex::builder::Building`][] 这类直接写入的实现之外, 也可
//! 以把本 trait 实现成装饰器.

use super::error::Result;

/// 顶点数据的写入接口.
///
/// 全部写入方法都以 [`Result`][] 返回 —— 失败通常是「当前元素与调用方法
/// 不匹配」, 由 [`crate::vertex::Error::WrongElement`][] 表达.
///
/// # 实现者注意
///
/// - **必须**维护「当前元素」状态, 并与 `VertexFormat` 的元素顺序同步.
/// - `end_vertex` 必须校验当前顶点已经写满.
/// - `default_color` / `unset_default_color` 只改状态, 不立即写出数据.
pub trait VertexConsumer {
	// ================ 必须实现的方法 ================

	/// 写入一个顶点的位置.
	///
	/// # 参数
	///
	/// - `x`, `y`, `z`: 位置分量, 坐标系与单位由上层约定.
	///
	/// # 错误
	///
	/// 当前元素不是 `Position` 用途, 或不是 `Float x3` 组合时, 返回
	/// [`crate::vertex::Error::WrongElement`][].
	fn vertex(&mut self, x: f32, y: f32, z: f32) -> Result<()>;

	/// 写入一个顶点的颜色.
	///
	/// # 参数
	///
	/// - `r`, `g`, `b`, `a`: 取值为 0 到 255 的分量. 存储时通常归一化到
	///   `[0, 1]`, 具体取决于格式中的元素类型 (`UByte` 会自动归一化).
	///
	/// # 错误
	///
	/// - 当前元素不是 `Color` 用途时, 返回
	///   [`crate::vertex::Error::WrongElement`][].
	/// - 已经设置默认颜色时, 返回
	///   [`crate::vertex::Error::DefaultColorActive`][].
	fn color(&mut self, r: u8, g: u8, b: u8, a: u8) -> Result<()>;

	/// 写入基础 UV 坐标 (层号为 0 的 `uv0`).
	///
	/// # 错误
	///
	/// 当前元素不是 `Uv` 用途, 或层号不为 0, 或不是 `Float x2` 组合时,
	/// 返回 [`crate::vertex::Error::WrongElement`][].
	fn uv(&mut self, u: f32, v: f32) -> Result<()>;

	/// 写入覆盖层坐标 (层号为 1 的 `uv1`).
	///
	/// 实体渲染中的「受伤闪红」等覆盖层效果会用到这组坐标.
	fn overlay_coords(&mut self, u: i32, v: i32) -> Result<()>;

	/// 写入光照坐标 (层号为 2 的 `uv2`).
	///
	/// 坐标按 `block_light | (sky_light << 16)` 打包, 两个分量各占 16 位.
	fn uv2(&mut self, u: i32, v: i32) -> Result<()>;

	/// 写入一个法线向量.
	///
	/// # 参数
	///
	/// - `x`, `y`, `z`: 单位向量的分量. 实现通常先 clamp 到 `[-1, 1]`,
	///   再按格式声明的类型 (一般是 `Byte`) 归一化存储.
	fn normal(&mut self, x: f32, y: f32, z: f32) -> Result<()>;

	/// 写入 4 个骨骼索引.
	///
	/// # 参数
	///
	/// - `indices`: 4 个骨骼在实例骨骼数组中的相对索引. 未用到的
	///   分量填 0.
	///
	/// # 错误
	///
	/// 当前元素不是 `BoneIndex` 用途, 或不是 `UByte x4` 组合时, 返回
	/// [`crate::vertex::Error::WrongElement`][].
	///
	/// # 与 `bone_weight` 的关系
	///
	/// 两个方法成对使用 —— `weight[i]` 是 `index[i]` 所指骨骼的影响
	/// 权重. 调用顺序由格式声明决定, 通常是 `bone_index` 在前.
	fn bone_index(&mut self, indices: [u8; 4]) -> Result<()>;

	/// 写入 4 个骨骼权重.
	///
	/// # 参数
	///
	/// - `weights`: 4 个骨骼的权重, 值域为 `[0, 255]`. 归一化到 `[0, 1]`
	///   后送入着色器. **各分量之和应为 255** —— 不做强制校验, 但
	///   未归一化的输入会让结果偏亮或偏暗.
	///
	/// # 错误
	///
	/// 当前元素不是 `BoneWeight` 用途, 或不是 `UByte x4` 组合时, 返回
	/// [`crate::vertex::Error::WrongElement`][].
	fn bone_weight(&mut self, weights: [u8; 4]) -> Result<()>;

	/// 结束并提交当前顶点.
	///
	/// 调用后, 实现者应校验当前顶点的所有元素都已写满, 并把写入位置
	/// 推进到下一个顶点的起始处.
	///
	/// # 错误
	///
	/// 当前顶点的元素尚未写满时, 返回
	/// [`crate::vertex::Error::IncompleteVertex`][].
	fn end_vertex(&mut self) -> Result<()>;

	/// 开启默认颜色.
	///
	/// 设置之后, 格式中的 `Color` 元素由实现者在推进时自动填充, 调用方
	/// 无需再调 [`Self::color`][]. 若调用方仍显式调用 `color`, 返回
	/// [`crate::vertex::Error::DefaultColorActive`][].
	fn default_color(&mut self, r: u8, g: u8, b: u8, a: u8);

	/// 清除默认颜色, 恢复由调用方显式写入 `Color` 的模式.
	fn unset_default_color(&mut self);

	// ================ 便捷方法 ================

	/// 用取值落在 `[0.0, 1.0]` 区间的浮点数写颜色.
	fn color_f32(&mut self, r: f32, g: f32, b: f32, a: f32) -> Result<()> {
		let to_u8 = |x: f32| (x.clamp(0.0, 1.0) * 255.0) as u8;
		self.color(to_u8(r), to_u8(g), to_u8(b), to_u8(a))
	}

	/// 写颜色, 输入是按 ARGB 顺序打包进 `u32` 的取值.
	fn color_argb(&mut self, argb: u32) -> Result<()> {
		let a = ((argb >> 24) & 0xFF) as u8;
		let r = ((argb >> 16) & 0xFF) as u8;
		let g = ((argb >> 8) & 0xFF) as u8;
		let b = (argb & 0xFF) as u8;
		self.color(r, g, b, a)
	}

	/// 一次性写入一个完整顶点 (不含骨骼数据).
	#[allow(clippy::too_many_arguments)]
	fn vertex_full(
		&mut self,
		x: f32, y: f32, z: f32,
		r: u8, g: u8, b: u8, a: u8,
		u: f32, v: f32,
		overlay: i32,
		light: i32,
		nx: f32, ny: f32, nz: f32,
	) -> Result<()> {
		self.vertex(x, y, z)?;
		self.color(r, g, b, a)?;
		self.uv(u, v)?;
		self.overlay_coords(overlay & 0xFFFF, (overlay >> 16) & 0xFFFF)?;
		self.uv2(light & 0xFFFF, (light >> 16) & 0xFFFF)?;
		self.normal(nx, ny, nz)?;
		self.end_vertex()
	}

	/// 一次性写入一个完整顶点 (含骨骼数据).
	///
	/// # 调用顺序
	///
	/// 与 [`Self::vertex_full`][] 相同, 只是在写完法线之后还要补上骨骼
	/// 索引与权重. 顺序**必须**与格式声明保持一致.
	#[allow(clippy::too_many_arguments)]
	fn vertex_full_skinned(
		&mut self,
		x: f32, y: f32, z: f32,
		r: u8, g: u8, b: u8, a: u8,
		u: f32, v: f32,
		overlay: i32,
		light: i32,
		nx: f32, ny: f32, nz: f32,
		bone_indices: [u8; 4],
		bone_weights: [u8; 4],
	) -> Result<()> {
		self.vertex(x, y, z)?;
		self.color(r, g, b, a)?;
		self.uv(u, v)?;
		self.overlay_coords(overlay & 0xFFFF, (overlay >> 16) & 0xFFFF)?;
		self.uv2(light & 0xFFFF, (light >> 16) & 0xFFFF)?;
		self.normal(nx, ny, nz)?;
		self.bone_index(bone_indices)?;
		self.bone_weight(bone_weights)?;
		self.end_vertex()
	}
}

// ================ 多路广播装饰器 ================

/// 把写入广播给多个下游的顶点消费者.
///
/// 每个写入方法都会转发给全部下游; 只要任一下游返回错误就立即中断,
/// 已写入的下游状态由各下游自行负责. 调用方需要保证各下游状态一致,
/// 或者在出错后丢弃整批数据.
pub struct MultiConsumer {
	downstreams: Vec<Box<dyn VertexConsumer>>,
}

impl MultiConsumer {
	/// 由一组下游消费者构造.
	///
	/// # Panics
	///
	/// 传入空集合时会 panic. 需要可失败的版本请改用 [`Self::try_new`][].
	pub fn new(downstreams: Vec<Box<dyn VertexConsumer>>) -> Self {
		assert!(
			!downstreams.is_empty(),
			"MultiConsumer requires at least one downstream"
		);
		Self { downstreams }
	}

	/// 可失败的构造: 下游集合为空时返回 [`None`][].
	pub fn try_new(downstreams: Vec<Box<dyn VertexConsumer>>) -> Option<Self> {
		if downstreams.is_empty() {
			None
		} else {
			Some(Self { downstreams })
		}
	}

	/// 下游消费者的数量.
	pub fn len(&self) -> usize {
		self.downstreams.len()
	}

	/// 是否没有任何下游.
	///
	/// 通过 [`Self::new`][] 或 [`Self::try_new`][] 构造出来的实例恒为
	/// `false`.
	pub fn is_empty(&self) -> bool {
		self.downstreams.is_empty()
	}
}

macro_rules! forward_fallible {
	($self:ident, $method:ident ( $( $arg:expr ),* )) => {{
		for d in &mut $self.downstreams {
			d.$method( $( $arg ),* )?;
		}
		Ok(())
	}};
}

impl VertexConsumer for MultiConsumer {
	fn vertex(&mut self, x: f32, y: f32, z: f32) -> Result<()> {
		forward_fallible!(self, vertex(x, y, z))
	}

	fn color(&mut self, r: u8, g: u8, b: u8, a: u8) -> Result<()> {
		forward_fallible!(self, color(r, g, b, a))
	}

	fn uv(&mut self, u: f32, v: f32) -> Result<()> {
		forward_fallible!(self, uv(u, v))
	}

	fn overlay_coords(&mut self, u: i32, v: i32) -> Result<()> {
		forward_fallible!(self, overlay_coords(u, v))
	}

	fn uv2(&mut self, u: i32, v: i32) -> Result<()> {
		forward_fallible!(self, uv2(u, v))
	}

	fn normal(&mut self, x: f32, y: f32, z: f32) -> Result<()> {
		forward_fallible!(self, normal(x, y, z))
	}

	fn bone_index(&mut self, indices: [u8; 4]) -> Result<()> {
		forward_fallible!(self, bone_index(indices))
	}

	fn bone_weight(&mut self, weights: [u8; 4]) -> Result<()> {
		forward_fallible!(self, bone_weight(weights))
	}

	fn end_vertex(&mut self) -> Result<()> {
		forward_fallible!(self, end_vertex())
	}

	fn default_color(&mut self, r: u8, g: u8, b: u8, a: u8) {
		for d in &mut self.downstreams {
			d.default_color(r, g, b, a);
		}
	}

	fn unset_default_color(&mut self) {
		for d in &mut self.downstreams {
			d.unset_default_color();
		}
	}
}
