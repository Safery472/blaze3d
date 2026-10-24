// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 顶点元素的定义.
//!
//! 一个顶点元素由数据类型, 用途, 索引与分量数共同描述, 本模块同时给出
//! 它到 [`wgpu::VertexFormat`][] 的映射.
//!
//! # 构造约定
//!
//! 合法的构造路径共有两条:
//!
//! - [`VertexFormatElement::new`][]: 通用构造. 非 UV 用途的 `index` 必须
//!   为 0, 否则返回 [`Error::InvalidIndex`][].
//! - 各便捷构造函数 (`position_f32x3` 等): 直接字面构造, 不会失败.
//!
//! 所有字段都是私有的, 外部无法绕过上述入口拼出元素.

use super::error::{Error, Result};

/// 元素分量的标量类型.
///
/// 它决定单个分量占几个字节, 以及在着色器中如何解释.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
	/// 32 位浮点数.
	Float,
	/// 8 位无符号整数, 常用于颜色与填充.
	UByte,
	/// 8 位有符号整数, 常用于法线.
	Byte,
	/// 16 位无符号整数, 常用于光照坐标.
	UShort,
	/// 16 位有符号整数, 常用于覆盖层坐标.
	Short,
	/// 无符号的 32 位整数.
	UInt,
	/// 有符号的 32 位整数.
	Int,
}

impl ElementType {
	/// 单个分量占用的字节数.
	///
	/// 这里算的只是「标量宽度」, 与 wgpu 的 `VertexFormat` 无关; 分量
	/// 数由 [`VertexFormatElement::count`][] 单独给出.
	pub const fn size(self) -> u32 {
		match self {
			Self::Float | Self::UInt | Self::Int => 4,
			Self::UByte | Self::Byte => 1,
			Self::UShort | Self::Short => 2,
		}
	}

	/// 供调试与日志使用的类型名.
	///
	/// **只供人阅读**. 不要用它做逻辑判断或字符串匹配 - 名字可能随
	/// 文档调整而变化, 需要稳定标识时请直接使用枚举变体.
	pub const fn name(self) -> &'static str {
		match self {
			Self::Float => "Float",
			Self::UByte => "Unsigned Byte",
			Self::Byte => "Byte",
			Self::UShort => "Unsigned Short",
			Self::Short => "Short",
			Self::UInt => "Unsigned Int",
			Self::Int => "Int",
		}
	}
}

/// 顶点属性的语义用途.
///
/// 着色器依靠用途与索引来定位属性 (`shader_location` 由格式分配).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementUsage {
	/// 顶点位置, 每个格式中只能有一个.
	Position,
	/// 法线向量, 每个格式中只能有一个.
	Normal,
	/// 顶点颜色, 每个格式中只能有一个.
	Color,
	/// 纹理坐标, 可以按索引分层 (`uv0` / `uv1` / `uv2`).
	///
	/// **这是唯一允许重复出现的用途**: 不同 UV 层靠 `index` 区分.
	/// 其余用途在 `VertexFormat::new` 中一旦重复就会被拒绝.
	Uv,
	/// 对齐填充. 只占一个槽位, 不会映射到 wgpu 属性.
	Padding,
	/// 通用属性, 语义由使用者自行约定.
	Generic,
	/// 骨骼索引. 由 4 个 `UByte` 组成, 每个分量是骨骼在实例骨骼数组
	/// 中的相对索引.
	///
	/// # 用途
	///
	/// 用于蒙皮: 顶点着色器据此从骨骼 storage buffer 中取出矩阵.
	///
	/// # 格式约束
	///
	/// 只接受 `UByte x4`. 它与 [`Self::BoneWeight`][] 成对出现 ——
	/// 前者存索引, 后者存权重.
	BoneIndex,
	/// 骨骼权重. 由 4 个 `UByte` 组成, 归一化到 `[0, 1]`.
	///
	/// # 用途
	///
	/// 与 [`Self::BoneIndex`][] 逐项对应 —— `weight[i]` 就是
	/// `index[i]` 所指骨骼的影响权重.
	///
	/// # 归一化
	///
	/// 各分量之和应为 1.0 (即四个 `UByte` 加起来等于 255). 这一点
	/// **不做强制**: 未归一化的输入会按给定值加权求和, 结果可能偏亮
	/// 或偏暗, 需要调用方自行保证.
	///
	/// # 格式约束
	///
	/// 只接受 `UByte x4`, 归一化由 wgpu 自动完成 (`Unorm8x4`).
	BoneWeight,
}

impl ElementUsage {
	/// 供调试与日志使用的用途名.
	///
	/// 与 [`ElementType::name`][] 一样只供人阅读, 不要用于逻辑判断.
	pub const fn name(self) -> &'static str {
		match self {
			Self::Position => "Position",
			Self::Normal => "Normal",
			Self::Color => "Vertex Color",
			Self::Uv => "UV",
			Self::Padding => "Padding",
			Self::Generic => "Generic",
			Self::BoneIndex => "Bone Index",
			Self::BoneWeight => "Bone Weight",
		}
	}
}

/// 单个顶点元素.
///
/// 它描述单个属性在顶点缓冲中的布局: 用途, 用途索引, 标量类型, 分量数.
///
/// # 字段可见性
///
/// 所有字段都是私有的. 外部只能通过 [`Self::new`][] 或便捷构造函数
/// 创建.
///
/// # 相等性
///
/// [`PartialEq`][] 与 [`Hash`][] 只比较 `(elem_type, usage, index,
/// count)`, 与元素名无关 - 元素名由 [`super::format::VertexFormat`][]
/// 单独管理.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexFormatElement {
	// 这四个字段共同决定元素的内存布局与语义, 缺一不可.
	// 它们全部私有: 外部无法绕过构造函数拼出非法组合 (例如非 UV
	// 用途携带非零 `index`).
	elem_type: ElementType,
	usage: ElementUsage,
	index: u32,
	count: u32,
}

impl VertexFormatElement {
	/// 通用构造入口.
	///
	/// # 行为
	///
	/// 非 UV 用途的 `index` 必须为 0; UV 用途的 `index` 表示 UV 层号,
	/// 取 0 或正整数皆可.
	///
	/// # 错误
	///
	/// 非 UV 用途传入了非零 `index` 时, 返回 [`Error::InvalidIndex`][].
	pub fn new(
		index: u32,
		elem_type: ElementType,
		usage: ElementUsage,
		count: u32,
	) -> Result<Self> {
		if index != 0 && !matches!(usage, ElementUsage::Uv) {
			return Err(Error::InvalidIndex { usage, index });
		}
		Ok(Self { elem_type, usage, index, count })
	}

	// ================ 便捷构造 ================
	//
	// 下列构造函数直接以字面量构造, 不经过 `new`.
	// 原因: 参数组合在编译期已经固定, 无需运行时校验;
	// 写成 `Self { ... }` 既能保持 const fn, 也避免了 unwrap.

	/// `Position` 用途, `Float x2`.
	pub const fn position_f32x2() -> Self {
		Self { elem_type: ElementType::Float, usage: ElementUsage::Position, index: 0, count: 2 }
	}
	/// `Position` 用途, `Float x3`.
	pub const fn position_f32x3() -> Self {
		Self { elem_type: ElementType::Float, usage: ElementUsage::Position, index: 0, count: 3 }
	}
	/// `Position` 用途, `Float x4`.
	pub const fn position_f32x4() -> Self {
		Self { elem_type: ElementType::Float, usage: ElementUsage::Position, index: 0, count: 4 }
	}

	/// `Color` 用途, `UByte x4`; 归一化到 `[0, 1]` 后送入着色器.
	pub const fn color_u8x4() -> Self {
		Self { elem_type: ElementType::UByte, usage: ElementUsage::Color, index: 0, count: 4 }
	}
	/// `Color` 用途, `Float x4`.
	pub const fn color_f32x4() -> Self {
		Self { elem_type: ElementType::Float, usage: ElementUsage::Color, index: 0, count: 4 }
	}

	/// `Normal` 用途, `Byte x3`, 归一化到 `[-1, 1]`.
	pub const fn normal_i8x3() -> Self {
		Self { elem_type: ElementType::Byte, usage: ElementUsage::Normal, index: 0, count: 3 }
	}

	/// `Padding` 用途, `UByte x1`; 不映射到 wgpu 属性.
	pub const fn padding_u8x1() -> Self {
		Self { elem_type: ElementType::UByte, usage: ElementUsage::Padding, index: 0, count: 1 }
	}

	/// `UV` 用途, 层号 0, `Float x2`.
	pub const fn uv0_f32x2() -> Self {
		Self { elem_type: ElementType::Float, usage: ElementUsage::Uv, index: 0, count: 2 }
	}
	/// `UV` 用途, 层号 1, `Short x2`.
	pub const fn uv1_i16x2() -> Self {
		Self { elem_type: ElementType::Short, usage: ElementUsage::Uv, index: 1, count: 2 }
	}
	/// `UV` 用途, 层号 2, `UShort x2`.
	pub const fn uv2_u16x2() -> Self {
		Self { elem_type: ElementType::UShort, usage: ElementUsage::Uv, index: 2, count: 2 }
	}

	/// `BoneIndex` 用途, `UByte x4`.
	///
	/// 4 个分量是骨骼在实例骨骼数组中的相对索引 —— 着色器用
	/// `instance.bone_offset + index[i]` 定位实际使用的矩阵.
	///
	/// 映射到 wgpu 的 `Uint8x4`, 每个分量的值域是 `[0, 255]`. 未用到
	/// 的分量填 0.
	pub const fn bone_index_u8x4() -> Self {
		Self {
			elem_type: ElementType::UByte,
			usage: ElementUsage::BoneIndex,
			index: 0,
			count: 4,
		}
	}

	/// `BoneWeight` 用途, `UByte x4`, 归一化到 `[0, 1]`.
	///
	/// 4 个分量与 [`Self::bone_index_u8x4`][] 逐项对应. 各分量之和应为
	/// 1.0 (即四个 `UByte` 加起来等于 255).
	///
	/// 映射到 wgpu 的 `Unorm8x4`, 归一化由硬件自动完成.
	pub const fn bone_weight_u8x4() -> Self {
		Self {
			elem_type: ElementType::UByte,
			usage: ElementUsage::BoneWeight,
			index: 0,
			count: 4,
		}
	}

	// ================ 字段读取 ================

	/// 取元素的标量类型.
	pub const fn elem_type(self) -> ElementType {
		self.elem_type
	}

	/// 取元素的语义用途.
	pub const fn usage(self) -> ElementUsage {
		self.usage
	}

	/// 取元素的分量数.
	pub const fn count(self) -> u32 {
		self.count
	}

	/// 元素携带的索引.
	///
	/// 通用元素恒为 0, UV 元素则是 UV 层号. 需要区分两者时应优先用
	/// [`Self::uv_index`][].
	pub const fn index(self) -> u32 {
		self.index
	}

	/// UV 层号.
	///
	/// 非 UV 元素返回 `None`.
	pub const fn uv_index(self) -> Option<u32> {
		match self.usage {
			ElementUsage::Uv => Some(self.index),
			_ => None,
		}
	}

	/// 元素占用的字节数, 等于 `elem_type.size() * count`.
	pub const fn byte_size(self) -> u32 {
		self.elem_type.size() * self.count
	}

	/// 判断是否属于 `Position` 用途.
	pub const fn is_position(self) -> bool {
		matches!(self.usage, ElementUsage::Position)
	}

	/// 判断是否属于 `Padding` 用途.
	pub const fn is_padding(self) -> bool {
		matches!(self.usage, ElementUsage::Padding)
	}

	// ================ 到 wgpu 的映射 ================

	/// 把元素映射为 wgpu 的顶点格式.
	///
	/// # 返回值
	///
	/// - `Ok(Some(fmt))`: 普通属性, 需要在管线里声明.
	/// - `Ok(None)`: `Padding` 元素, 不上传, 也不占 `shader_location`.
	/// - `Err(Error::UnsupportedElement)`: 该组合在 wgpu 中没有对应项.
	///
	/// # 支持的组合
	///
	/// | 用途         | 类型     | 分量 | wgpu 格式     |
	/// |--------------|----------|------|---------------|
	/// | `Position`   | `Float`  | 2    | `Float32x2`   |
	/// | `Position`   | `Float`  | 3    | `Float32x3`   |
	/// | `Position`   | `Float`  | 4    | `Float32x4`   |
	/// | `Color`      | `UByte`  | 4    | `Unorm8x4`    |
	/// | `Color`      | `Float`  | 4    | `Float32x4`   |
	/// | `UV`         | `Float`  | 2    | `Float32x2`   |
	/// | `UV`         | `Short`  | 2    | `Sint16x2`    |
	/// | `UV`         | `UShort` | 2    | `Uint16x2`    |
	/// | `Normal`     | `Byte`   | 3    | `Snorm8x4`    |
	/// | `Normal`     | `Float`  | 3    | `Float32x3`   |
	/// | `BoneIndex`  | `UByte`  | 4    | `Uint8x4`     |
	/// | `BoneWeight` | `UByte`  | 4    | `Unorm8x4`    |
	/// | `Padding`    | 任意     | 任意 | `None`        |
	pub fn wgpu_format(self) -> Result<Option<wgpu::VertexFormat>> {
		use wgpu::VertexFormat as VF;
		use ElementType as T;
		use ElementUsage as U;

		let fmt = match (self.usage, self.elem_type, self.count) {
			(U::Position, T::Float, 2) => VF::Float32x2,
			(U::Position, T::Float, 3) => VF::Float32x3,
			(U::Position, T::Float, 4) => VF::Float32x4,

			(U::Color, T::UByte, 4) => VF::Unorm8x4,
			(U::Color, T::Float, 4) => VF::Float32x4,

			(U::Uv, T::Float, 2) => VF::Float32x2,
			(U::Uv, T::Short, 2) => VF::Sint16x2,
			(U::Uv, T::UShort, 2) => VF::Uint16x2,

			(U::Normal, T::Byte, 3) => VF::Snorm8x4,
			(U::Normal, T::Float, 3) => VF::Float32x3,

			// 骨骼索引映射到 `Uint8x4`, 不做归一化.
			(U::BoneIndex, T::UByte, 4) => VF::Uint8x4,

			// 骨骼权重映射到 `Unorm8x4`, 归一化到 `[0, 1]`.
			(U::BoneWeight, T::UByte, 4) => VF::Unorm8x4,

			// `Padding` 是合法元素, 但不会上传.
			(U::Padding, _, _) => return Ok(None),

			// 兜底分支: 该组合在 wgpu 里找不到对应项.
			_ => return Err(Error::UnsupportedElement { element: self }),
		};
		Ok(Some(fmt))
	}
}
