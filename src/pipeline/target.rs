// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 离屏渲染目标.
//!
//! [`RenderTarget`] 描述一次渲染的目标表面: 颜色纹理, 可选的深度纹理,
//! 可选的 MSAA, 视口尺寸, 以及清屏参数.
//!
//! # `Target` trait
//!
//! [`Target`] 抽象出「可以作为渲染目标的东西」. 离屏目标的实现是
//! [`RenderTarget`], 窗口表面的实现是 [`crate::pipeline::SurfaceFrame`].
//!
//! # 尺寸与视口
//!
//! `width` / `height` 是纹理尺寸; `view_width` / `view_height` 是渲染时
//! 生效的视口范围. 两者通常相同.
//!
//! # MSAA
//!
//! `sample_count == 1` 表示没有 MSAA, 颜色纹理直接充当渲染目标.
//!
//! `sample_count > 1` 表示启用 MSAA, 内部会创建:
//!
//! - MSAA 颜色纹理 (`sample_count`).
//! - 解析颜色纹理 (`sample_count = 1`) —— 对外暴露为 `color_texture`.
//! - MSAA 深度纹理 (`sample_count`, 若启用了深度).
//!
//! 渲染时管线输出到 MSAA 纹理, 再由 wgpu 自动解析到 `color_texture`.

use super::error::{Error, Result};
use crate::state::{ColorFormat, DepthFormat};

// ================ Target trait ================

/// 能充当渲染目标的类型.
///
/// # 实现者
///
/// - [`RenderTarget`] —— 离屏目标, 持有长期纹理.
/// - [`crate::pipeline::SurfaceFrame`] —— 窗口表面, 纹理每帧变化.
pub trait Target {
	/// 本次渲染使用的颜色附件.
	fn color_attachment(&self) -> wgpu::RenderPassColorAttachment<'_>;

	/// 深度附件; 返回 `None` 表示没有深度.
	fn depth_attachment(&self) -> Option<wgpu::RenderPassDepthStencilAttachment<'_>>;

	/// 视口的宽高.
	fn viewport_size(&self) -> (u32, u32);

	/// 颜色附件的格式.
	fn color_format(&self) -> ColorFormat;
}

// ================ 尺寸辅助 ================

/// 尺寸回退时使用的默认宽度.
pub const DEFAULT_WIDTH: u32 = 854;
/// 尺寸回退时使用的默认高度.
pub const DEFAULT_HEIGHT: u32 = 480;

/// 把请求尺寸限制到设备支持的范围内.
///
/// 宽高都非零且不超过设备上限时原样返回; 否则退回 [`DEFAULT_WIDTH`] /
/// [`DEFAULT_HEIGHT`].
pub fn clamp_to_limits(device: &wgpu::Device, width: u32, height: u32) -> (u32, u32) {
	let limit = device.limits().max_texture_dimension_2d;
	if width > 0 && width <= limit && height > 0 && height <= limit {
		(width, height)
	} else {
		(DEFAULT_WIDTH, DEFAULT_HEIGHT)
	}
}

// ================ 渲染目标 ================

/// 离屏渲染目标 —— 自己持有颜色与深度纹理.
///
/// # MSAA
///
/// `sample_count == 1` 时没有 MSAA. `sample_count > 1` 时:
///
/// - `color_texture` / `color_view` 指向**解析目标** (`sample_count = 1`).
/// - `msaa_color_texture` / `msaa_color_view` 是 MSAA 纹理.
/// - `color_attachment()` 以 MSAA 视图为主附件, 解析目标作
///   `resolve_target`.
///
/// 读回使用 `color_texture` —— 那里面是解析后的结果.
pub struct RenderTarget {
	// ---- 尺寸与视口 ----

	/// 纹理宽度 (像素).
	pub width: u32,
	/// 纹理高度 (像素).
	pub height: u32,
	/// 视口宽度 (像素).
	pub view_width: u32,
	/// 视口高度 (像素).
	pub view_height: u32,

	// ---- 格式参数 ----

	/// 颜色纹理的格式.
	pub color_format: ColorFormat,
	/// 深度纹理的格式; 为 `None` 时不创建深度附件.
	pub depth_format: Option<DepthFormat>,
	/// 采样数; 取 1 表示关闭 MSAA.
	pub sample_count: u32,

	// ---- 纹理附件 ----

	/// 输出颜色所用的纹理.
	///
	/// 启用 MSAA 时它保存的是解析后的结果 —— 采样, 读回与 present 都
	/// 走这张纹理.
	pub color_texture: wgpu::Texture,
	/// 颜色纹理的视图.
	pub color_view: wgpu::TextureView,

	/// MSAA 颜色纹理; 未启用 MSAA 时为 `None`.
	pub msaa_color_texture: Option<wgpu::Texture>,
	/// MSAA 颜色纹理的视图; 未启用 MSAA 时为 `None`.
	pub msaa_color_view: Option<wgpu::TextureView>,

	/// 深度纹理; `depth_format` 为 `None` 时它也是 `None`.
	pub depth_texture: Option<wgpu::Texture>,
	/// 深度纹理的视图.
	pub depth_view: Option<wgpu::TextureView>,

	// ---- 清屏参数 ----

	/// 清屏颜色, 按 RGBA 顺序.
	pub clear_color: [f32; 4],
	/// 清屏时写入的深度值.
	pub clear_depth: f32,
}

impl RenderTarget {
	/// 创建不带 MSAA 的渲染目标.
	///
	/// 等价于调用 [`Self::new_msaa`] 并传 `sample_count = 1`.
	pub fn new(
		device: &wgpu::Device,
		width: u32,
		height: u32,
		color_format: ColorFormat,
		depth_format: Option<DepthFormat>,
		label: Option<&str>,
	) -> Result<Self> {
		Self::new_msaa(device, width, height, color_format, depth_format, 1, label)
	}

	/// 创建渲染目标, 可指定 MSAA 采样数.
	///
	/// # 参数
	///
	/// - `sample_count` —— 采样数; 1 表示关闭 MSAA, >= 2 表示启用.
	///
	/// # 错误
	///
	/// - 尺寸为 0 —— [`Error::ZeroSize`].
	/// - 尺寸超出设备上限 —— [`Error::SizeExceedsLimit`].
	/// - `sample_count == 0` —— [`Error::InvalidSampleCount`].
	pub fn new_msaa(
		device: &wgpu::Device,
		width: u32,
		height: u32,
		color_format: ColorFormat,
		depth_format: Option<DepthFormat>,
		sample_count: u32,
		label: Option<&str>,
	) -> Result<Self> {
		if width == 0 || height == 0 {
			return Err(Error::ZeroSize { width, height });
		}
		// 1 表示关闭 MSAA, >= 2 表示启用; 0 没有意义 —— 判定口径与
		// `DepthBuffer::new` 保持一致.
		//
		// 真正合法的取值 (1 / 2 / 4 / 8 / 16) 由 wgpu 在
		// `create_texture` 时校验 —— 这里只拦明显非法的 0.
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

		let size = wgpu::Extent3d {
			width,
			height,
			depth_or_array_layers: 1,
		};

		// 解析纹理 (未启用 MSAA 时它就是颜色纹理); `sample_count` 恒为 1.
		let color_texture = device.create_texture(&wgpu::TextureDescriptor {
			label,
			size,
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: color_format.wgpu(),
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT
				| wgpu::TextureUsages::TEXTURE_BINDING
				| wgpu::TextureUsages::COPY_SRC,
			view_formats: &[],
		});
		let color_view = color_texture.create_view(&Default::default());

		// MSAA 颜色纹理, 仅在 `sample_count > 1` 时创建.
		let (msaa_color_texture, msaa_color_view) = if sample_count > 1 {
			let tex = device.create_texture(&wgpu::TextureDescriptor {
				label,
				size,
				mip_level_count: 1,
				sample_count,
				dimension: wgpu::TextureDimension::D2,
				format: color_format.wgpu(),
				usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
				view_formats: &[],
			});
			let view = tex.create_view(&Default::default());
			(Some(tex), Some(view))
		} else {
			(None, None)
		};

		// 深度纹理 —— 其 `sample_count` 与 MSAA 保持一致.
		let (depth_texture, depth_view) = if let Some(df) = depth_format {
			let tex = device.create_texture(&wgpu::TextureDescriptor {
				label,
				size,
				mip_level_count: 1,
				sample_count,
				dimension: wgpu::TextureDimension::D2,
				format: df.wgpu(),
				usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
				view_formats: &[],
			});
			let view = tex.create_view(&Default::default());
			(Some(tex), Some(view))
		} else {
			(None, None)
		};

		Ok(Self {
			width,
			height,
			view_width: width,
			view_height: height,
			color_format,
			depth_format,
			sample_count,
			color_texture,
			color_view,
			msaa_color_texture,
			msaa_color_view,
			depth_texture,
			depth_view,
			clear_color: [0.0, 0.0, 0.0, 0.0],
			clear_depth: 1.0,
		})
	}

	/// 修改视口尺寸 (纹理尺寸保持不变).
	pub fn set_view_size(&mut self, width: u32, height: u32) {
		self.view_width = width;
		self.view_height = height;
	}

	/// 按目标尺寸重建.
	///
	/// `clear_color` / `clear_depth` / `sample_count` 都会保留.
	pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) -> Result<()> {
		if self.width == width && self.height == height {
			return Ok(());
		}

		let clear_color = self.clear_color;
		let clear_depth = self.clear_depth;

		let mut new = Self::new_msaa(
			device,
			width,
			height,
			self.color_format,
			self.depth_format,
			self.sample_count,
			Some("render_target"),
		)?;

		new.clear_color = clear_color;
		new.clear_depth = clear_depth;

		*self = new;
		Ok(())
	}

	/// 判断是否创建了深度附件.
	pub const fn use_depth(&self) -> bool {
		self.depth_format.is_some()
	}

	/// 判断是否启用了 MSAA.
	pub const fn use_msaa(&self) -> bool {
		self.sample_count > 1
	}
}

impl Target for RenderTarget {
	/// 构造本次渲染的颜色附件.
	///
	/// 启用 MSAA 时以 MSAA 视图为主附件, `color_view` (解析目标) 作
	/// `resolve_target`.
	fn color_attachment(&self) -> wgpu::RenderPassColorAttachment<'_> {
		let ops = wgpu::Operations {
			load: wgpu::LoadOp::Clear(wgpu::Color {
				r: self.clear_color[0] as f64,
				g: self.clear_color[1] as f64,
				b: self.clear_color[2] as f64,
				a: self.clear_color[3] as f64,
			}),
			store: wgpu::StoreOp::Store,
		};

		if let Some(msaa_view) = &self.msaa_color_view {
			wgpu::RenderPassColorAttachment {
				view: msaa_view,
				resolve_target: Some(&self.color_view),
				depth_slice: None,
				ops,
			}
		} else {
			wgpu::RenderPassColorAttachment {
				view: &self.color_view,
				resolve_target: None,
				depth_slice: None,
				ops,
			}
		}
	}

	fn depth_attachment(&self) -> Option<wgpu::RenderPassDepthStencilAttachment<'_>> {
		self.depth_view
			.as_ref()
			.map(|view| wgpu::RenderPassDepthStencilAttachment {
				view,
				depth_ops: Some(wgpu::Operations {
					load: wgpu::LoadOp::Clear(self.clear_depth),
					store: wgpu::StoreOp::Store,
				}),
				stencil_ops: None,
			})
	}

	fn viewport_size(&self) -> (u32, u32) {
		(self.view_width, self.view_height)
	}

	fn color_format(&self) -> ColorFormat {
		self.color_format
	}
}
