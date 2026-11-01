// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 顶点格式的定义.
//!
//! [`VertexFormat`][] 描述顶点数据在缓冲中的完整布局: 一串有序的
//! [`VertexFormatElement`][], 每个元素在顶点内的字节偏移, 以及顶点的
//! stride.
//!
//! # 构造约定
//!
//! 唯一的合法构造路径是 [`VertexFormat::new`][]. 构造时会依次完成下列
//! 校验:
//!
//! - 元素名互不重复.
//! - 非 UV 用途的 usage 唯一; UV 用途按 `uv_index` 判重.
//! - 每个元素的 `count` 都不为 0.
//! - 每个元素都能映射到 [`wgpu::VertexFormat`][] (`Padding` 例外).
//! - 第一个元素不是 `Padding`.
//! - 每个元素的字节数可表示, stride 累加不会溢出 `u32`.
//! - stride 是 4 字节的倍数 —— `(Normal, Byte, 3)` 在 wgpu 侧是
//!   `Snorm8x4`, 占 4 字节槽位.
//!
//! 任一项校验失败都会返回 [`Error`][], 不会 panic.
//!
//! # stride 对齐
//!
//! wgpu 要求 `array_stride` 是 [`wgpu::VERTEX_ALIGNMENT`] (4 字节) 的倍数.
//! `(Normal, Byte, 3)` 在 wgpu 侧是 `Snorm8x4`, 按 4 字节计入; 其余组合
//! 需要补齐时在格式里加 `Padding`, 否则构造会返回
//! [`Error::StrideMisaligned`][].
//!
//! # 相等性
//!
//! [`PartialEq`][] 与 [`Hash`][] 只比较 `vertex_size` 与元素序列, 与元素
//! 名无关.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::element::{ElementUsage, VertexFormatElement};
use super::error::{Error, Result};

/// 一份顶点格式.
///
/// 只能通过 [`Self::new`][] 构造; 构造完成后不可变, 可以跨线程共享
/// ([`SharedVertexFormat`][]).
///
/// **元素顺序即写入顺序** —— [`crate::vertex::VertexConsumer`][] 的实现
/// 沿该顺序推进, 一旦声明顺序与调用顺序不一致, 就会返回
/// [`Error::WrongElement`][].
#[derive(Debug, Clone)]
pub struct VertexFormat {
	elements: Vec<VertexFormatElement>,
	names: Vec<String>,
	name_to_index: HashMap<String, usize>,
	offsets: Vec<u32>,
	vertex_size: u32,
	wgpu_attributes: Vec<wgpu::VertexAttribute>,
}

impl VertexFormat {
	/// 由带名字的元素序列构造一份格式.
	///
	/// 元素的先后顺序就是 [`crate::vertex::VertexConsumer`][] 的写入顺序,
	/// 名字则供调试与 [`Self::index_of_name`][] 使用.
	///
	/// # Errors
	///
	/// 校验规则见模块文档, 任意一项不通过都会返回错误, 本方法不会
	/// panic:
	///
	/// - [`Error::Empty`][]: 元素序列是空的.
	/// - [`Error::DuplicateName`][]: 有元素重名.
	/// - [`Error::DuplicateUsage`][]: 同一 usage 重复, UV 按 `uv_index`
	///   判重.
	/// - [`Error::LeadingPadding`][]: 第一个元素是 `Padding`.
	/// - [`Error::ZeroSizedElement`][]: 某个元素的 `count` 为 0.
	/// - [`Error::UnsupportedElement`][]: 元素映射不到 wgpu 顶点格式.
	/// - [`Error::StrideOverflow`][]: 单个元素的字节数或 stride 累加结果
	///   超出 `u32`.
	/// - [`Error::StrideMisaligned`][]: stride 不是 4 的倍数.
	pub fn new<I, S>(named_elements: I) -> Result<Self>
	where
		I: IntoIterator<Item = (S, VertexFormatElement)>,
		S: Into<String>,
	{
		let mut elements = Vec::new();
		let mut names = Vec::new();
		let mut name_to_index = HashMap::new();
		let mut offsets = Vec::new();
		let mut offset: u32 = 0;
		let mut seen_usages: Vec<(ElementUsage, Option<u32>)> = Vec::new();

		for (name, element) in named_elements {
			let name = name.into();

			if name_to_index.contains_key(&name) {
				return Err(Error::DuplicateName { name });
			}

			if elements.is_empty() && element.is_padding() {
				return Err(Error::LeadingPadding);
			}

			let key = (element.usage(), element.uv_index());
			if element.usage() != ElementUsage::Uv {
				if seen_usages.iter().any(|(usage, _)| *usage == element.usage()) {
					return Err(Error::DuplicateUsage { usage: element.usage() });
				}
			} else if seen_usages.contains(&key) {
				return Err(Error::DuplicateUsage { usage: ElementUsage::Uv });
			}
			seen_usages.push(key);

			if element.count() == 0 {
				return Err(Error::ZeroSizedElement { element });
			}

			element.wgpu_format()?;

			let size = element.checked_byte_size().ok_or(Error::StrideOverflow)?;
			let new_offset = offset.checked_add(size).ok_or(Error::StrideOverflow)?;

			offsets.push(offset);
			offset = new_offset;

			name_to_index.insert(name.clone(), elements.len());
			names.push(name);
			elements.push(element);
		}

		if elements.is_empty() {
			return Err(Error::Empty);
		}

		// wgpu 要求顶点缓冲的 stride 是 `VERTEX_ALIGNMENT` (4 字节) 的倍数.
		// 在这里拦下, 调用方才拿得到 `Err`, 而不是等到创建管线时才收到
		// wgpu 的验证错误.
		if !offset.is_multiple_of(wgpu::VERTEX_ALIGNMENT as u32) {
			return Err(Error::StrideMisaligned { stride: offset });
		}

		let mut wgpu_attributes = Vec::new();
		let mut shader_location = 0u32;
		for (i, element) in elements.iter().enumerate() {
			if let Some(format) = element.wgpu_format()? {
				wgpu_attributes.push(wgpu::VertexAttribute {
					format,
					offset: offsets[i] as wgpu::BufferAddress,
					shader_location,
				});
				shader_location += 1;
			}
		}

		Ok(Self {
			elements,
			names,
			name_to_index,
			offsets,
			vertex_size: offset,
			wgpu_attributes,
		})
	}

	/// 一个顶点占用的字节数, 即 stride.
	pub const fn vertex_size(&self) -> u32 {
		self.vertex_size
	}

	/// 一个顶点占用的 `u32` 字数, 即 `vertex_size / 4` 向下取整.
	///
	/// 把顶点数据当作 `u32` / `f32` 数组来索引时, 用它作为步长. 若
	/// stride 不是 4 的倍数, 该结果是取整后的值, 按字索引会错位.
	pub const fn integer_size(&self) -> u32 {
		self.vertex_size / 4
	}

	/// 全部元素, 顺序就是写入顺序.
	pub fn elements(&self) -> &[VertexFormatElement] {
		&self.elements
	}

	/// 全部元素名, 与 [`Self::elements`][] 逐项对应.
	pub fn element_names(&self) -> &[String] {
		&self.names
	}

	/// 第 `index` 个元素在顶点内的字节偏移.
	///
	/// `index` 越界时返回 `None`.
	pub fn offset_of(&self, index: usize) -> Option<u32> {
		self.offsets.get(index).copied()
	}

	/// 按名字查找元素的下标.
	///
	/// 名字不存在时返回 `None`.
	pub fn index_of_name(&self, name: &str) -> Option<usize> {
		self.name_to_index.get(name).copied()
	}

	/// 构造供 wgpu 使用的顶点缓冲布局.
	///
	/// 返回值借用了本格式持有的属性数组, 因此它的生命周期不会超过
	/// `self`.
	pub fn wgpu_layout(&self) -> wgpu::VertexBufferLayout<'_> {
		wgpu::VertexBufferLayout {
			array_stride: self.vertex_size as wgpu::BufferAddress,
			step_mode: wgpu::VertexStepMode::Vertex,
			attributes: &self.wgpu_attributes,
		}
	}
}

impl PartialEq for VertexFormat {
	fn eq(&self, other: &Self) -> bool {
		self.vertex_size == other.vertex_size && self.elements == other.elements
	}
}

impl Eq for VertexFormat {}

impl Hash for VertexFormat {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.vertex_size.hash(state);
		for element in &self.elements {
			element.hash(state);
		}
	}
}

/// 可跨线程共享的顶点格式句柄.
pub type SharedVertexFormat = Arc<VertexFormat>;

// ================ 预设格式 ================

/// 常用格式的预设集合.
pub mod defaults {
	use super::VertexFormat;
	use crate::vertex::element::VertexFormatElement as E;

	fn make<const N: usize>(entries: [(&'static str, E); N]) -> VertexFormat {
		VertexFormat::new(entries).expect("default format must be valid")
	}

	/// 只含位置属性.
	pub fn position_only() -> VertexFormat {
		make([("Position", E::position_f32x3())])
	}

	/// 位置与颜色.
	pub fn position_color() -> VertexFormat {
		make([
			("Position", E::position_f32x3()),
			("Color", E::color_u8x4()),
		])
	}

	/// 位置与 UV0.
	pub fn position_tex() -> VertexFormat {
		make([
			("Position", E::position_f32x3()),
			("UV0", E::uv0_f32x2()),
		])
	}

	/// 位置, 颜色与 UV0.
	pub fn position_color_tex() -> VertexFormat {
		make([
			("Position", E::position_f32x3()),
			("Color", E::color_u8x4()),
			("UV0", E::uv0_f32x2()),
		])
	}

	/// 位置, 颜色与光照坐标.
	pub fn position_color_lightmap() -> VertexFormat {
		make([
			("Position", E::position_f32x3()),
			("Color", E::color_u8x4()),
			("UV2", E::uv2_u16x2()),
		])
	}

	/// 方块渲染常用: 位置, 颜色, UV0, 光照与法线, 每个顶点 36 字节.
	///
	/// # 布局
	///
	/// | 元素       | 偏移 | 字节 |
	/// |------------|------|------|
	/// | `Position` | 0    | 12   |
	/// | `Color`    | 12   | 4    |
	/// | `UV0`      | 16   | 8    |
	/// | `UV2`      | 24   | 4    |
	/// | `Normal`   | 28   | 4    |
	/// | `Padding`  | 32   | 4    |
	///
	/// `Normal` 在 wgpu 侧是 `Snorm8x4`, 占满 4 字节槽位 (第 4 个分量不
	/// 参与运算), 元素因此止于 32; 末尾 4 字节 `Padding` 把 stride 补到
	/// 36. **不要**把它换成 `u8 x1`: `VertexFormat::new` 会返回
	/// [`Error::StrideMisaligned`][crate::vertex::Error::StrideMisaligned],
	/// 而 `make` 内部直接 panic.
	pub fn block() -> VertexFormat {
		make([
			("Position", E::position_f32x3()),
			("Color", E::color_u8x4()),
			("UV0", E::uv0_f32x2()),
			("UV2", E::uv2_u16x2()),
			("Normal", E::normal_i8x3()),
			("Padding", E::padding_u8x4()),
		])
	}

	/// 实体渲染常用: 在 `block` 的基础上多一个覆盖层 UV1, 每个顶点 40 字节.
	///
	/// # 布局
	///
	/// | 元素       | 偏移 | 字节 |
	/// |------------|------|------|
	/// | `Position` | 0    | 12   |
	/// | `Color`    | 12   | 4    |
	/// | `UV0`      | 16   | 8    |
	/// | `UV1`      | 24   | 4    |
	/// | `UV2`      | 28   | 4    |
	/// | `Normal`   | 32   | 4    |
	/// | `Padding`  | 36   | 4    |
	///
	/// 与 `block` 相同, 末尾 `Padding` 负责把 36 补齐到 40.
	pub fn new_entity() -> VertexFormat {
		make([
			("Position", E::position_f32x3()),
			("Color", E::color_u8x4()),
			("UV0", E::uv0_f32x2()),
			("UV1", E::uv1_i16x2()),
			("UV2", E::uv2_u16x2()),
			("Normal", E::normal_i8x3()),
			("Padding", E::padding_u8x4()),
		])
	}

	/// 蒙皮渲染用的格式: 位置, 颜色, 骨骼索引与骨骼权重.
	///
	/// # 布局
	///
	/// 每个顶点占 24 字节:
	///
	/// - `Position` 为 `Float x3` (12 字节).
	/// - `Color` 为 `UByte x4` (4 字节).
	/// - `BoneIndex` 为 `UByte x4` (4 字节).
	/// - `BoneWeight` 为 `UByte x4` (4 字节).
	///
	/// # 与 `new_entity` 的差异
	///
	/// 不含 UV / 光照 / 法线 —— 它是最小可用的蒙皮格式, 适合验证与
	/// 简单场景. 需要完整的实体渲染时, 套用同样的骨骼元素自行组装
	/// 格式即可.
	pub fn position_color_skinned() -> VertexFormat {
		make([
			("Position", E::position_f32x3()),
			("Color", E::color_u8x4()),
			("BoneIndex", E::bone_index_u8x4()),
			("BoneWeight", E::bone_weight_u8x4()),
		])
	}
}
