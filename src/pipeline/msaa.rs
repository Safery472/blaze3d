// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! MSAA 颜色缓冲.
//!
//! [`MsaaBuffer`] 是多重采样的颜色纹理, 用于:
//!
//! - 为 [`crate::pipeline::SurfaceFrame`] 提供 MSAA —— surface 纹理的
//!   采样数固定为 1, 不能直接作为 MSAA 目标;
//! - 为需要 MSAA 的离屏渲染提供单独的 MSAA 目标.
//!
//! # 工作方式
//!
//! MSAA 颜色缓冲不直接作为最终输出. 渲染时:
//!
//! 1. 管线渲染到 MSAA 纹理 (`sample_count` > 1).
//! 2. wgpu 自动把 MSAA 纹理**解析**到目标纹理 (`sample_count` = 1).
//! 3. 目标纹理用作最终输出 (present / 后续采样).
//!
//! `SurfaceFrame` 内部处理这个流程: MSAA 纹理的视图作为主颜色附件,
//! surface 纹理的视图作为 `resolve_target`.
//!
//! # 采样数
//!
//! 常见取值:
//!
//! - 4: 平衡选项 —— 画质明显改善, 性能开销适中, **推荐作为默认值**.
//! - 8: 更平滑, 开销更高.
//! - 2: 效果较弱, 开销最低.
//!
//! 不同后端支持的最大采样数不同; 4x 几乎在所有桌面 GPU 上可用.
//!
//! # 与 `DepthBuffer` 的配合
//!
//! 启用 MSAA 时, 深度缓冲的 `sample_count` 必须与 MSAA 缓冲一致, wgpu
//! 在创建 render pass 时会校验这一点.

use super::error::{Error, Result};
use crate::state::ColorFormat;

/// MSAA 颜色缓冲.
///
/// # 生命周期
///
/// 持有 [`wgpu::Texture`] 与 [`wgpu::TextureView`], 随 `MsaaBuffer`
/// 一起 drop.
pub struct MsaaBuffer {
	texture: wgpu::Texture,
	view: wgpu::TextureView,
	format: ColorFormat,
	sample_count: u32,
	width: u32,
	height: u32,
}

impl MsaaBuffer {
	/// 创建 MSAA 颜色缓冲.
	///
	/// # 参数
	///
	/// - `sample_count`: 采样数. **必须 >= 2** —— 传 1 没有意义, 应当走
	///   非 MSAA 路径.
	///
	/// # 错误
	///
	/// - 尺寸为 0: [`Error::ZeroSize`].
	/// - 尺寸超限: [`Error::SizeExceedsLimit`].
	/// - `sample_count` < 2: [`Error::InvalidSampleCount`].
	pub fn new(
		device: &wgpu::Device,
		width: u32,
		height: u32,
		format: ColorFormat,
		sample_count: u32,
	) -> Result<Self> {
		if width == 0 || height == 0 {
			return Err(Error::ZeroSize { width, height });
		}
		if sample_count < 2 {
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
			label: Some("msaa_buffer"),
			size: wgpu::Extent3d {
				width,
				height,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count,
			dimension: wgpu::TextureDimension::D2,
			format: format.wgpu(),
			// MSAA 纹理只作为渲染附件. 解析到 resolve 目标后, 由 resolve
			// 目标承担采样 / present 的职责.
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

	/// 返回颜色格式.
	pub fn format(&self) -> ColorFormat {
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
