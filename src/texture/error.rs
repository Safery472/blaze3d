// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 纹理与绑定模块的错误类型.
//!
//! 本模块覆盖 [`crate::texture`] 中所有可能失败的公开 API. 与 `vertex` /
//! `pipeline` / `system` 相同, 这里只暴露一个错误枚举 —— 调用方匹配这
//! 一个枚举即可.
//!
//! # 不包含的内容
//!
//! - **`wgpu` 底层错误**: `wgpu` 的 `create_*` 要么成功要么 panic, 这里
//!   不再额外包装.
//! - **`BindGroup` 条目与 `layout` 不匹配**: 该项检查由 `wgpu` 在创建时
//!   完成, 报错信息已经足够清晰, 无需二次封装.

use thiserror::Error;

use super::image::TextureKind;

/// 纹理与绑定操作可能返回的错误.
///
/// 枚举标注了 `#[non_exhaustive]`, 因此下游的 `match` 必须保留 `_` 分支,
/// 以应对后续版本新增的变体.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum Error {
	// ================ 纹理相关 ================

	/// 纹理的尺寸为零.
	///
	/// 在 [`crate::texture::Texture::new`] 与
	/// [`crate::texture::Texture::from_bytes`] 中返回. `wgpu` 拒绝任何
	/// 宽高或深度为 0 的纹理.
	#[error("texture size must be non-zero (got {width}x{height}x{depth})")]
	ZeroSize {
		/// 调用方请求的宽度.
		width: u32,
		/// 调用方请求的高度.
		height: u32,
		/// 请求的深度, 在 2D 与 Cube 下恒为 1.
		depth: u32,
	},

	/// 请求的尺寸超出设备上限.
	///
	/// 由 [`crate::texture::Texture::new`] 返回.
	///
	/// # 上限的来源
	///
	/// - `D2` / `Cube`: 取 `max_texture_dimension_2d`.
	/// - `D3`: 取 `max_texture_dimension_3d`.
	#[error("texture size {width}x{height}x{depth} exceeds device limit {limit}")]
	SizeExceedsLimit {
		/// 调用方请求的宽度.
		width: u32,
		/// 调用方请求的高度.
		height: u32,
		/// 请求的深度, 在 2D 与 Cube 下恒为 1.
		depth: u32,
		/// 设备允许的维度上限.
		limit: u32,
	},

	/// 纹理维度与传入的参数不兼容.
	///
	/// 在 [`crate::texture::Texture::new`] 与
	/// [`crate::texture::Texture::generate_mipmaps`] 中返回.
	///
	/// 典型的触发场景:
	///
	/// - MSAA 仅可用于 `D2`.
	/// - Mipmap 生成目前只实现了 `D2`.
	#[error("texture kind {kind:?} is incompatible: {reason}")]
	IncompatibleKind {
		/// 触发该错误的纹理维度.
		kind: TextureKind,
		/// 判定为不兼容的原因.
		reason: &'static str,
	},

	/// 该纹理格式不支持上传数据.
	///
	/// 由 [`crate::texture::Texture::from_bytes`] 返回. 压缩格式与深度
	/// 模板格式等都无法直接从 CPU 端上传.
	#[error("texture format {format:?} does not support data upload")]
	UnsupportedFormat {
		/// 触发该错误的纹理格式.
		format: wgpu::TextureFormat,
	},

	/// 上传数据的大小与纹理尺寸不符.
	///
	/// 由 [`crate::texture::Texture::from_bytes`] 返回. 期望大小按
	/// `width * height * layers * block_size` 计算, 其中 `layers` 在
	/// `D3` 下取 `depth`, 在 `Cube` 下取 6.
	#[error("data size mismatch: expected {expected} bytes, got {actual}")]
	DataSizeMismatch {
		/// 该纹理所需的字节数.
		expected: u64,
		/// 实际传入的字节数.
		actual: u64,
	},

	/// 暂不支持一次上传多层 Mipmap 数据.
	///
	/// 由 [`crate::texture::Texture::from_bytes`] 返回. 需要多 Mip 层的
	/// 纹理时, 应改用 [`crate::texture::Texture::new`] 创建, 再通过
	/// `queue.write_texture` 按层写入.
	#[error("data upload with mip_level_count > 1 is not supported (requested {requested})")]
	UnsupportedMipLevels {
		/// 调用方请求的 Mip 层数.
		requested: u32,
	},

	/// MSAA 纹理不接受数据上传.
	///
	/// 由 [`crate::texture::Texture::from_bytes`] 返回. MSAA 纹理只能
	/// 充当 `render pass` 的附件 —— `wgpu` 不允许向 `sample_count > 1`
	/// 的纹理写入来自 CPU 的数据.
	///
	/// # 与 `UnsupportedMipLevels` 的区别
	///
	/// 两者的触发点都在 `from_bytes`, 成因却不同: 前者源于层数, 后者
	/// 源于采样数. 拆成两个变体, 调用方可以直接看出是哪个字段导致.
	#[error("data upload with sample_count > 1 is not supported (requested {requested})")]
	UnsupportedSampleCount {
		/// 调用方请求的采样数.
		requested: u32,
	},

	// ================ Mipmap 相关 ================

	/// 纹理没有 Mipmap 层, 因此无从生成.
	///
	/// 由 [`crate::texture::Texture::generate_mipmaps`] 返回. 当
	/// `mip_level_count == 1` 时, 并不存在可生成的级别.
	#[error("texture has no mip levels (mip_level_count = 1)")]
	NoMipLevels,

	/// 纹理缺少生成 Mipmap 所必需的用途.
	///
	/// 由 [`crate::texture::Texture::generate_mipmaps`] 返回. 生成
	/// Mipmap 需要下列用途:
	///
	/// - `TEXTURE_BINDING` —— 用于采样上一级.
	/// - `STORAGE_BINDING` (`compute` 路径) 或 `RENDER_ATTACHMENT`
	///   (`render` 回退路径).
	#[error("mipmap generation requires TEXTURE_BINDING and (STORAGE_BINDING or RENDER_ATTACHMENT) usage")]
	MissingMipmapUsage,
}

/// `texture` 模块使用的 `Result` 别名.
///
/// 便于把函数签名简写成 `Result<T>`. 需要时也可以用
/// `Result<T, OtherError>` 指定别的错误类型.
pub type Result<T, E = Error> = std::result::Result<T, E>;
