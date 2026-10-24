// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 采样器资源的封装.
//!
//! [`Sampler`] 封装了 wgpu 的采样器. 它决定纹理采样时的过滤方式,
//! 寻址模式与 Mipmap 选择等.
//!
//! # 与纹理的关系
//!
//! 采样器**不**绑定到纹理 —— 它们是彼此独立的资源, 在 `bind group` 里
//! 分别绑定. 同一个采样器可以用于多张纹理, 同一张纹理也可以搭配多个
//! 采样器 (例如同一张图在不同材质里使用不同过滤).
//!
//! # 预设
//!
//! 常用的预设共三个:
//!
//! - [`Sampler::nearest`] —— 最近邻过滤. 适用于像素画, 像素完美缩放.
//! - [`Sampler::linear`] —— 双线性过滤. 适用于一般纹理, 放大时平滑.
//! - [`Sampler::linear_mipmap`] —— 三线性过滤. 缩小时也平滑, 要求纹理
//!   带 Mipmap.
//!
//! 阴影贴图用 [`Sampler::comparison`] —— 采样时由硬件比较深度值, 返回
//! `[0, 1]` 范围内的阴影因子.
//!
//! 需要其他组合时, 用 [`Sampler::new`] 传入自定义描述.
//!
//! # 与 `Deref` 的关系
//!
//! `Sampler` 实现了 `Deref<Target = wgpu::Sampler>`, 因此可以直接当作
//! [`wgpu::Sampler`] 使用.

use std::ops::Deref;

/// 纹理采样时使用的采样器.
///
/// # 寻址模式
///
/// 所有预设都使用 `ClampToEdge` —— 采样坐标超出 `[0, 1]` 时钳制到边界.
/// 需要重复平铺 (`Repeat`) 或镜像 (`MirrorRepeat`) 时, 用 [`Self::new`]
/// 自定义.
///
/// # Mipmap
///
/// [`Self::linear_mipmap`] 使用三线性过滤; 其余预设用 `Nearest`, 或者
/// 根本不在意 Mip. 纹理没有 Mip 时, Mipmap 过滤方式不影响采样结果.
///
/// # `FilterMode` vs `MipmapFilterMode`
///
/// wgpu 30 起 `mag_filter` / `min_filter` 使用 [`wgpu::FilterMode`], 而
/// `mipmap_filter` 使用 [`wgpu::MipmapFilterMode`] —— 两个枚举的变体名
/// 相同 (`Nearest` / `Linear`) 但类型不同, 不可互换.
pub struct Sampler {
	inner: wgpu::Sampler,
}

impl Sampler {
	/// 构造最近邻过滤的采样器.
	///
	/// # 用途
	///
	/// - 像素画与复古风格.
	/// - 数据纹理 (法线贴图, 查找表) —— 需要精确的像素值.
	pub fn nearest(device: &wgpu::Device, label: Option<&str>) -> Self {
		Self::preset(
			device,
			label,
			wgpu::FilterMode::Nearest,
			wgpu::FilterMode::Nearest,
			wgpu::MipmapFilterMode::Nearest,
		)
	}

	/// 构造双线性过滤的采样器.
	///
	/// # 用途
	///
	/// - 常规纹理 (颜色贴图, 字体).
	/// - 放大时能平滑过渡.
	///
	/// 缩小时不会自动选择合适的 Mip 层 —— 需要三线性过滤时改用
	/// [`Self::linear_mipmap`].
	pub fn linear(device: &wgpu::Device, label: Option<&str>) -> Self {
		Self::preset(
			device,
			label,
			wgpu::FilterMode::Linear,
			wgpu::FilterMode::Linear,
			wgpu::MipmapFilterMode::Nearest,
		)
	}

	/// 构造三线性过滤 (线性 + Mipmap) 的采样器.
	///
	/// # 用途
	///
	/// - 已经带 Mipmap 的纹理.
	/// - 缩小时自动选 Mip 层, 并在相邻层间插值.
	///
	/// # 前置条件
	///
	/// 若纹理没有 Mipmap, 效果与 [`Self::linear`] 完全一致.
	pub fn linear_mipmap(device: &wgpu::Device, label: Option<&str>) -> Self {
		Self::preset(
			device,
			label,
			wgpu::FilterMode::Linear,
			wgpu::FilterMode::Linear,
			wgpu::MipmapFilterMode::Linear,
		)
	}

	/// 比较采样器 —— 专供阴影贴图.
	///
	/// # 行为
	///
	/// 采样时硬件自动比较纹理值与参考值, 返回 `[0, 1]` 范围内的阴影因子:
	///
	/// - 未被遮挡 (纹理值 `<=` 参考值) → 1.0.
	/// - 被遮挡 (纹理值 `>` 参考值) → 0.0.
	///
	/// 使用 `Linear` 过滤时, 硬件会做 2x2 PCF —— 边缘柔和, 无需在着色器里
	/// 手动多次采样.
	///
	/// # WGSL 声明
	///
	/// **比较采样器在 WGSL 里是 `sampler_comparison`**, 且只能通过
	/// `textureSampleCompare` 采样 —— **不要**改用 `textureSample`:
	///
	/// ```wgsl
	/// @group(3) @binding(0) var shadow_tex: texture_depth_2d;
	/// @group(3) @binding(1) var shadow_sampler: sampler_comparison;
	///
	/// let shadow = textureSampleCompare(shadow_tex, shadow_sampler, uv, ref_depth);
	/// ```
	///
	/// `bind group layout` 里对应 `SamplerBindingType::Comparison` —— 用
	/// [`crate::texture::BindGroupLayoutBuilder::sampler_comparison`] 声明.
	///
	/// # 参数
	///
	/// - **`compare`**: 比较函数. 阴影贴图常用
	///   [`wgpu::CompareFunction::LessEqual`].
	pub fn comparison(
		device: &wgpu::Device,
		label: Option<&str>,
		compare: wgpu::CompareFunction,
	) -> Self {
		Self {
			inner: device.create_sampler(&wgpu::SamplerDescriptor {
				label,
				address_mode_u: wgpu::AddressMode::ClampToEdge,
				address_mode_v: wgpu::AddressMode::ClampToEdge,
				address_mode_w: wgpu::AddressMode::ClampToEdge,
				mag_filter: wgpu::FilterMode::Linear,
				min_filter: wgpu::FilterMode::Linear,
				mipmap_filter: wgpu::MipmapFilterMode::Nearest,
				compare: Some(compare),
				..Default::default()
			}),
		}
	}

	/// 按自定义描述创建采样器.
	///
	/// # 用途
	///
	/// 需要以下特性时使用:
	///
	/// - `Repeat` / `MirrorRepeat` 等非 `ClampToEdge` 寻址.
	/// - 需要各向异性过滤 (`anisotropy_clamp`).
	/// - 自定义比较函数, 或不想用预设的 `Linear` 过滤.
	pub fn new(device: &wgpu::Device, desc: wgpu::SamplerDescriptor<'_>) -> Self {
		Self {
			inner: device.create_sampler(&desc),
		}
	}

	/// 返回内部的 wgpu 采样器引用.
	pub fn raw(&self) -> &wgpu::Sampler {
		&self.inner
	}

	/// 三个预设共用的构造入口.
	///
	/// # 固定值
	///
	/// - 三个轴的寻址模式都固定为 `ClampToEdge`.
	/// - `lod_min_clamp` / `lod_max_clamp` / `compare` /
	///   `anisotropy_clamp` 取 [`wgpu::SamplerDescriptor`] 的默认值.
	///
	/// # 可变值
	///
	/// - **`mag_filter`**: 放大时的过滤方式.
	/// - **`min_filter`**: 缩小时的过滤方式.
	/// - **`mipmap_filter`**: Mip 层之间的过滤方式.
	///
	/// # 类型差异
	///
	/// `mag_filter` / `min_filter` 的类型是 [`wgpu::FilterMode`], 而
	/// `mipmap_filter` 的类型是 [`wgpu::MipmapFilterMode`].
	///
	/// # 为什么私有
	///
	/// 公开它会给出一个「寻址模式固定」的接口, 与 [`Self::new`] 的通用性
	/// 重叠. 预设方法已覆盖常见场景 —— 需要其他组合时, 调用方直接用
	/// [`Self::new`] 传入完整描述.
	fn preset(
		device: &wgpu::Device,
		label: Option<&str>,
		mag_filter: wgpu::FilterMode,
		min_filter: wgpu::FilterMode,
		mipmap_filter: wgpu::MipmapFilterMode,
	) -> Self {
		Self {
			inner: device.create_sampler(&wgpu::SamplerDescriptor {
				label,
				address_mode_u: wgpu::AddressMode::ClampToEdge,
				address_mode_v: wgpu::AddressMode::ClampToEdge,
				address_mode_w: wgpu::AddressMode::ClampToEdge,
				mag_filter,
				min_filter,
				mipmap_filter,
				..Default::default()
			}),
		}
	}
}

impl Deref for Sampler {
	type Target = wgpu::Sampler;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}
