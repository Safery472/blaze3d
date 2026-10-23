//! 深度缓冲.
//!
//! [`DepthBuffer`] 是独立于颜色附件的深度纹理, 用于:
//!
//! - 为不携带深度的渲染目标 (例如 [`crate::pipeline::SurfaceFrame`])
//!   提供深度测试;
//! - 为多个需要共享深度的渲染目标提供统一深度.
//!
//! # 与 `RenderTarget` 的关系
//!
//! [`crate::pipeline::RenderTarget`] 内部管理自己的深度纹理;
//! `SurfaceFrame` 不携带深度, 由 [`DepthBuffer`] 单独提供.
//!
//! # 采样数
//!
//! 启用 MSAA 时, 深度缓冲的 `sample_count` 必须与 MSAA 颜色缓冲一致,
//! wgpu 会校验这一点.

use super::error::{Error, Result};
use crate::state::DepthFormat;

/// 独立的深度纹理.
///
/// # 生命周期
///
/// 持有 [`wgpu::Texture`] 与 [`wgpu::TextureView`], 随 `DepthBuffer`
/// 一起 drop.
pub struct DepthBuffer {
	texture: wgpu::Texture,
	view: wgpu::TextureView,
	format: DepthFormat,
	sample_count: u32,
	width: u32,
	height: u32,
}

impl DepthBuffer {
	/// 创建深度纹理.
	///
	/// # 参数
	///
	/// - `sample_count`: 采样数. 1 表示不启用 MSAA, >= 2 表示启用; 启用
	///   MSAA 时必须与颜色缓冲的采样数保持一致.
	///
	/// # 错误
	///
	/// - 尺寸为 0: [`Error::ZeroSize`].
	/// - 尺寸超限: [`Error::SizeExceedsLimit`].
	/// - `sample_count` 为 0: [`Error::InvalidSampleCount`].
	pub fn new(
		device: &wgpu::Device,
		width: u32,
		height: u32,
		format: DepthFormat,
		sample_count: u32,
	) -> Result<Self> {
		if width == 0 || height == 0 {
			return Err(Error::ZeroSize { width, height });
		}
		if sample_count == 0 {
			return Err(Error::InvalidSampleCount { count: sample_count });
		}
		let limit = device.limits().max_texture_dimension_2d;
		if width > limit || height > limit {
			return Err(Error::SizeExceedsLimit {
				requested_width: width,
				requested_height: height,
				limit,
			});
		}

		let texture = device.create_texture(&wgpu::TextureDescriptor {
			label: Some("depth_buffer"),
			size: wgpu::Extent3d {
				width,
				height,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count,
			dimension: wgpu::TextureDimension::D2,
			format: format.wgpu(),
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			view_formats: &[],
		});

		let view = texture.create_view(&Default::default());

		Ok(Self {
			texture,
			view,
			format,
			sample_count,
			width,
			height,
		})
	}

	/// 把缓冲重建为目标尺寸.
	///
	/// 尺寸与当前一致时为空操作; `format` 与 `sample_count` 保持不变.
	pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) -> Result<()> {
		if self.width == width && self.height == height {
			return Ok(());
		}
		*self = Self::new(device, width, height, self.format, self.sample_count)?;
		Ok(())
	}

	// ================ 访问器 ================

	/// 返回底层的 [`wgpu::Texture`].
	pub fn raw(&self) -> &wgpu::Texture {
		&self.texture
	}

	/// 返回默认的纹理视图.
	pub fn view(&self) -> &wgpu::TextureView {
		&self.view
	}

	/// 返回深度格式.
	pub fn format(&self) -> DepthFormat {
		self.format
	}

	/// 返回采样数.
	pub fn sample_count(&self) -> u32 {
		self.sample_count
	}

	/// 返回尺寸, 形式为 `(width, height)`.
	pub fn size(&self) -> (u32, u32) {
		(self.width, self.height)
	}

	/// 返回宽度 (像素).
	pub fn width(&self) -> u32 {
		self.width
	}

	/// 返回高度 (像素).
	pub fn height(&self) -> u32 {
		self.height
	}
}
