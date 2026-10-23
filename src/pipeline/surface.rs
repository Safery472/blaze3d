//! 窗口表面 —— 逐帧取纹理, 提交与配置.
//!
//! [`SurfaceFrame`] 抽象「一次窗口渲染」: 从 [`wgpu::Surface`] 取到当前帧
//! 的纹理, 当作渲染目标使用, 渲染结束后再由 [`SurfaceFrame::present`]
//! 提交给显示系统.
//!
//! # 与 `RenderTarget` 的差异
//!
//! - **纹理来源不同** —— [`crate::pipeline::RenderTarget`] 自己创建纹理
//!   并长期持有; `SurfaceFrame` 则每帧从 surface 取一张新纹理.
//! - **生命周期不同** —— `RenderTarget` 可跨多帧使用; `SurfaceFrame` 只
//!   对应一帧, 渲染完必须 `present`.
//! - **附件** —— `SurfaceFrame` 不带深度与 MSAA, 这两者由
//!   [`DepthBuffer`] 与 [`MsaaBuffer`] 单独提供, 传给
//!   [`SurfaceFrame::acquire`].
//!
//! # Drop 顺序
//!
//! [`SurfaceFrame::present`] 会显式解构 `self`: 先释放所有引用 surface
//! 纹理的视图, **再**把纹理归还给 swap chain. 一旦顺序颠倒 (view 比
//! texture 晚 drop), swap chain 就无法立即重用该图像, 结果是 GPU 内存
//! 持续增长.
//!
//! 这是 Rust 默认的 drop 顺序 (按字段声明顺序) 与 surface 生命周期之间
//! 的冲突 —— 只能手工控制.
//!
//! # wgpu 30 的变化
//!
//! wgpu 30 移除了 `SurfaceTexture::present()` —— 改用
//! [`wgpu::Queue::present`]. 本模块的 [`SurfaceFrame::present`] 内部就是
//! 调用 `queue.present(texture)`.
//!
//! [`wgpu::SurfaceConfiguration`] 新增了 `color_space` 字段, 用于 HDR /
//! wide-gamut 输出. 默认值 `SurfaceColorSpace::Auto` 适用于大多数场景.

use super::depth::DepthBuffer;
use super::error::{Error, Result};
use super::msaa::MsaaBuffer;
use super::target::Target;
use crate::state::{ColorFormat, DepthFormat};

/// 从窗口表面取得的一帧.
///
/// 由 [`SurfaceFrame::acquire`] 构造, 持有 surface 纹理及由它派生的视图;
/// 调用 [`SurfaceFrame::present`] 提交之后即不再可用.
pub struct SurfaceFrame {
	// 持有 surface 纹理 —— present 时要用到它.
	texture: wgpu::SurfaceTexture,

	// surface 纹理的视图.
	//
	// 未启用 MSAA 时它就是主颜色附件;
	// 启用 MSAA 时它充当解析目标 (`resolve_target`).
	view: wgpu::TextureView,

	// 深度视图的克隆 —— acquire 时从传入的深度缓冲复制.
	depth_view: Option<wgpu::TextureView>,
	// 深度格式, 跟随传入的深度缓冲.
	depth_format: Option<DepthFormat>,

	// MSAA 颜色视图的克隆 —— acquire 时从传入的 MSAA 缓冲复制.
	msaa_view: Option<wgpu::TextureView>,

	// 颜色格式与尺寸, 均取自 surface 配置.
	format: ColorFormat,
	width: u32,
	height: u32,
}

impl SurfaceFrame {
	/// 从 surface 取到当前帧.
	///
	/// # 参数
	///
	/// - `depth` —— 可选的深度缓冲; 传 `Some` 即启用深度测试.
	/// - `msaa` —— 可选的 MSAA 缓冲; 传 `Some` 即启用 MSAA.
	///
	/// # 尺寸一致性
	///
	/// `depth` 与 `msaa` 的尺寸必须与 `config` 一致 —— wgpu 会在创建
	/// render pass 时校验.
	///
	/// # 采样数一致性
	///
	/// `depth` 与 `msaa` 的 `sample_count` 必须相同, 也要与管线使用的
	/// `sample_count` 相同.
	///
	/// # wgpu 30
	///
	/// [`wgpu::Surface::get_current_texture`] 返回的是
	/// [`wgpu::CurrentSurfaceTexture`] 枚举, 共七个变体: `Success` /
	/// `Suboptimal` / `Timeout` / `Occluded` / `Outdated` / `Lost` /
	/// `Validation`.
	pub fn acquire(
		surface: &wgpu::Surface,
		config: &wgpu::SurfaceConfiguration,
		depth: Option<&DepthBuffer>,
		msaa: Option<&MsaaBuffer>,
	) -> Result<Self> {
		let texture = match surface.get_current_texture() {
			wgpu::CurrentSurfaceTexture::Success(t) => t,
			wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
			wgpu::CurrentSurfaceTexture::Timeout => return Err(Error::SurfaceTimeout),
			wgpu::CurrentSurfaceTexture::Occluded => return Err(Error::SurfaceOccluded),
			wgpu::CurrentSurfaceTexture::Outdated => return Err(Error::SurfaceOutdated),
			wgpu::CurrentSurfaceTexture::Lost => return Err(Error::SurfaceLost),
			wgpu::CurrentSurfaceTexture::Validation => return Err(Error::SurfaceValidation),
		};

		let view = texture.texture.create_view(&Default::default());

		let format = ColorFormat::from_wgpu(config.format)
			.ok_or(Error::UnsupportedSurfaceFormat { format: config.format })?;

		let (depth_view, depth_format) = match depth {
			Some(d) => (Some(d.view().clone()), Some(d.format())),
			None => (None, None),
		};

		let msaa_view = msaa.map(|m| m.view().clone());

		Ok(Self {
			texture,
			view,
			depth_view,
			depth_format,
			msaa_view,
			format,
			width: config.width,
			height: config.height,
		})
	}

	/// 把当前帧提交到显示系统.
	///
	/// # 行为
	///
	/// 显式解构 `self`, 让资源按正确顺序释放:
	///
	/// 1. **先 drop 所有引用 surface 纹理的视图** (`view` /
	///    `depth_view` / `msaa_view`).
	/// 2. **再 present 纹理** —— 把图像归还给 swap chain.
	///
	/// # 为什么不能依赖默认 drop 顺序
	///
	/// `SurfaceFrame` 的字段按声明顺序 drop —— `texture` 排在 `view` 前面.
	/// 而 `present` 把 `texture` 从 `self` 中移出之后, 其余字段 (`view`
	/// 等) 才被 drop. 若那时 texture 已经归还给 swap chain, 视图却仍在
	/// 引用它底层的图像, swap chain 就无法立即重用该图像, **最终导致
	/// GPU 内存持续增长**.
	///
	/// 显式解构能保证 view 在 present **之前**释放.
	///
	/// # wgpu 30
	///
	/// `SurfaceTexture::present()` 已被移除, 提交改由
	/// [`wgpu::Queue::present`] 完成.
	///
	/// # 参数
	///
	/// - `queue` —— 与 [`Self::acquire`] 所用设备配套的 queue.
	pub fn present(self, queue: &wgpu::Queue) {
		// 解构 self: 显式安排每个字段的释放时机.
		//
		// `SurfaceFrame` 没有实现 `Drop`, 所以可以把字段逐个 move 出来.
		let SurfaceFrame {
			texture,
			view,
			depth_view,
			msaa_view,
			depth_format: _,
			format: _,
			width: _,
			height: _,
		} = self;

		// **关键**: 先释放全部视图, 让 surface 纹理不再被任何视图引用.
		drop(view);
		drop(depth_view);
		drop(msaa_view);

		// **然后**再把纹理归还给 swap chain.
		queue.present(texture);
	}

	/// 返回当前帧的颜色格式.
	pub fn format(&self) -> ColorFormat {
		self.format
	}

	/// 返回当前帧的尺寸, 形如 `(width, height)`.
	pub fn size(&self) -> (u32, u32) {
		(self.width, self.height)
	}

	/// 返回深度格式; `None` 表示本帧没有深度附件.
	pub fn depth_format(&self) -> Option<DepthFormat> {
		self.depth_format
	}

	/// 返回本帧是否启用了 MSAA.
	pub fn use_msaa(&self) -> bool {
		self.msaa_view.is_some()
	}

	/// 返回底层的 surface 纹理.
	pub fn raw_texture(&self) -> &wgpu::Texture {
		&self.texture.texture
	}
}

impl Target for SurfaceFrame {
	/// 构造本次渲染的颜色附件.
	///
	/// 启用 MSAA 时以 MSAA 视图为主附件, surface 视图充当
	/// `resolve_target`.
	fn color_attachment(&self) -> wgpu::RenderPassColorAttachment<'_> {
		let ops = wgpu::Operations {
			load: wgpu::LoadOp::Clear(wgpu::Color {
				r: 0.0,
				g: 0.0,
				b: 0.0,
				a: 1.0,
			}),
			store: wgpu::StoreOp::Store,
		};

		if let Some(msaa_view) = &self.msaa_view {
			wgpu::RenderPassColorAttachment {
				view: msaa_view,
				resolve_target: Some(&self.view),
				depth_slice: None,
				ops,
			}
		} else {
			wgpu::RenderPassColorAttachment {
				view: &self.view,
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
					load: wgpu::LoadOp::Clear(1.0),
					store: wgpu::StoreOp::Store,
				}),
				stencil_ops: None,
			})
	}

	fn viewport_size(&self) -> (u32, u32) {
		(self.width, self.height)
	}

	fn color_format(&self) -> ColorFormat {
		self.format
	}
}

// ================ 表面配置 ================

/// 为窗口表面协商参数并完成配置.
///
/// 格式优先选 `sRGB`, present 模式优先选 `Fifo`.
///
/// # 与 [`reconfigure_surface`] 的分工
///
/// - 本函数 —— **初次**配置: 协商格式, present 模式与 alpha 模式.
/// - [`reconfigure_surface`] —— **运行时**调整尺寸, 复用已有配置.
///
/// 窗口创建后调用一次即可; 之后的 resize 交给 `reconfigure_surface`.
///
/// # wgpu 30
///
/// - `SurfaceConfiguration` 新增 `color_space` 字段 —— 用
///   [`wgpu::SurfaceColorSpace::Auto`] 让系统自行选择.
/// - `RequestAdapterOptions` 新增 `apply_limit_buckets` 字段 —— 调用
///   方在请求 adapter 时需要自行指定, 本函数不涉及 adapter 请求.
pub fn configure_surface(
	surface: &wgpu::Surface,
	adapter: &wgpu::Adapter,
	device: &wgpu::Device,
	width: u32,
	height: u32,
) -> Result<wgpu::SurfaceConfiguration> {
	let caps = surface.get_capabilities(adapter);

	if caps.formats.is_empty() {
		return Err(Error::NoSurfaceFormat);
	}

	let format = caps
		.formats
		.iter()
		.find(|f| {
			matches!(
				**f,
				wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Rgba8UnormSrgb
			)
		})
		.copied()
		.unwrap_or(caps.formats[0]);

	if ColorFormat::from_wgpu(format).is_none() {
		return Err(Error::UnsupportedSurfaceFormat { format });
	}

	let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
		wgpu::PresentMode::Fifo
	} else {
		caps.present_modes
			.first()
			.copied()
			.unwrap_or(wgpu::PresentMode::Fifo)
	};

	let alpha_mode = caps
		.alpha_modes
		.first()
		.copied()
		.unwrap_or(wgpu::CompositeAlphaMode::Auto);

	let config = wgpu::SurfaceConfiguration {
		usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
		format,
		color_space: wgpu::SurfaceColorSpace::Auto,
		width,
		height,
		present_mode,
		desired_maximum_frame_latency: 2,
		alpha_mode,
		view_formats: vec![],
	};

	surface.configure(device, &config);

	Ok(config)
}

/// 按新尺寸重新配置表面.
///
/// # 与 [`configure_surface`] 的分工
///
/// - [`configure_surface`] —— **初次**配置: 协商格式, present 模式与
///   alpha 模式.
/// - 本函数 —— **运行时**调整尺寸: 复用已有配置的格式与模式, 只更新
///   宽高.
///
/// 格式与 present 模式在初次配置时就已协商完毕, 窗口 resize 不会改变
/// 这些属性, 因此无需重走协商流程. 这也意味着本函数**不需要**
/// [`wgpu::Adapter`] —— 它与 `configure_surface` 的签名差异是有意为之.
///
/// # 行为
///
/// - 新尺寸任一为 0 —— 无操作, 返回 `None`. 这是窗口最小化时的常见
///   取值, 而 wgpu 不接受零尺寸配置.
/// - 新尺寸与 `config` 当前尺寸相同 —— 仍调用
///   [`wgpu::Surface::configure`] 刷新 (用于 `SurfaceOutdated` 的恢复
///   路径), 但返回 `None`, 调用方无需重建附件.
/// - 新尺寸不同 —— 更新 `config.width` / `config.height` 并调用
///   `configure`, 返回 `Some((width, height))`.
///
/// # 调用方职责
///
/// 返回 `Some` 时, 调用方**必须**重建所有与 surface 尺寸绑定的资源:
///
/// - 深度缓冲 ([`DepthBuffer::resize`]).
/// - MSAA 缓冲 ([`MsaaBuffer::resize`]).
/// - 相机视口 ([`crate::camera::PerspectiveCamera::set_viewport`] /
///   [`crate::camera::OrthographicCamera::set_viewport`] ——
///   `set_viewport` 不在 [`crate::camera::Camera`] trait 上, 见
///   `camera` 模块文档).
/// - 离屏渲染目标 ([`crate::pipeline::RenderTarget::resize`]).
///
/// 漏掉任何一项都会渲染出错 —— wgpu 在创建 render pass 时会校验附件
/// 尺寸, 尺寸不匹配就直接 panic.
///
/// # 为什么尺寸未变也要 configure
///
/// [`SurfaceFrame::acquire`] 可能返回 [`Error::SurfaceOutdated`] —— 表示
/// surface 配置与窗口不再匹配. 这种情形下尺寸未必变化 (某些平台切换
/// 窗口状态时并不改变尺寸), 但仍要重新 configure 才能恢复. 所以本函数
/// 不会因为「尺寸相同」就跳过 configure.
///
/// # 参数
///
/// - `surface` —— 目标表面.
/// - `device` —— 与初次配置时使用的设备相同.
/// - `config` —— 由 [`configure_surface`] 返回的配置, 原地更新.
/// - `width` / `height` —— 新的逻辑尺寸 (像素).
///
/// # 返回值
///
/// - `None` —— 尺寸为 0 或尺寸未变, 无需重建附件.
/// - `Some((width, height))` —— 尺寸已更新, 需要重建附件.
pub fn reconfigure_surface(
	surface: &wgpu::Surface,
	device: &wgpu::Device,
	config: &mut wgpu::SurfaceConfiguration,
	width: u32,
	height: u32,
) -> Option<(u32, u32)> {
	// 零尺寸 —— 窗口最小化. wgpu 不接受零尺寸配置, 因此保持原状态.
	// **不要**在这里调用 configure: 那会让 surface 进入非法状态.
	if width == 0 || height == 0 {
		return None;
	}

	let changed = config.width != width || config.height != height;

	config.width = width;
	config.height = height;

	// 尺寸未变时同样要调用 —— 这是 `SurfaceOutdated` 的恢复路径, 见本
	// 函数文档的「为什么尺寸未变也要 configure」一节.
	surface.configure(device, config);

	if changed {
		Some((width, height))
	} else {
		None
	}
}
