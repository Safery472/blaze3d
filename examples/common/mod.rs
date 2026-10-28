// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! example 共用的 GPU 资源与 winit 事件循环骨架.
//!
//! 本模块只提供通用接口: `Gfx` 封装 wgpu 初始化、surface、`RenderSystem`、
//! 相机、深度缓冲和 MSAA 缓冲; `Scene` 是 example 需要实现的最小 trait;
//! `Runner` 是通用事件循环. **不包含**任何具体 example 的配置项, 配置由
//! 各 example 自己定义.

use std::sync::Arc;
use std::time::Instant;

use blaze3d::camera::PerspectiveCamera;
use blaze3d::pipeline::{
	configure_surface, DepthBuffer, Error as PipelineError, MsaaBuffer, SurfaceFrame,
	DEFAULT_SURFACE_CLEAR_COLOR,
};
use blaze3d::prelude::*;
use blaze3d::system::{RenderSystem, RenderSystemDescriptor};
use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

// ================ 通用工具 ================

/// 从编译时环境变量读取 `f32`, 不存在或解析失败时回退到默认值.
///
/// `option_env!` 在编译时求值, 返回 `Option<&'static str>`. 字符串转
/// `f32` 是运行时操作, 所以这个函数**只在** `Scene::new` 里调用一次,
/// 结果存进 `Scene` 字段, 避免每帧重复解析.
///
/// 配置项的名称和默认值由各 example 自己决定, 本函数只负责解析.
pub fn read_f32(env: Option<&'static str>, default: f32) -> f32 {
	env.and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// sRGB 分量转线性空间, 输入和输出都在 `[0, 1]`.
///
/// `SurfaceFrame::set_clear_color` 把传入值按线性空间解释, 再由 wgpu
/// 编码成 sRGB 写入纹理 (见该方法的「颜色空间」一节). 设计稿和 GUI 里
/// 的颜色通常是 sRGB 值, 直接传给 `set_clear_color` 会偏亮, 需要先经
/// 本函数换算.
///
/// 例: sRGB `0.11` 灰换算出线性 `≈ 0.0116`, 写入 sRGB 表面后屏幕显示
/// 的就是 `0.11` 灰, 与 Bevy 默认背景一致.
pub fn srgb_to_linear(c: f32) -> f32 {
	if c <= 0.04045 {
		c / 12.92
	} else {
		((c + 0.055) / 1.055).powf(2.4)
	}
}

// ================ GPU 资源 ================

/// 所有 example 共用的 GPU 资源与窗口状态.
pub struct Gfx {
	pub window: Arc<Window>,
	pub surface: wgpu::Surface<'static>,
	pub device: Arc<wgpu::Device>,
	pub queue: Arc<wgpu::Queue>,
	pub config: wgpu::SurfaceConfiguration,
	pub window_format: ColorFormat,
	pub system: RenderSystem,
	pub camera: PerspectiveCamera,
	pub depth: DepthBuffer,
	pub msaa: MsaaBuffer,
	/// 本帧的清屏颜色, 按 RGBA 顺序, **线性空间**.
	///
	/// 每帧由 `acquire_frame` 通过 `SurfaceFrame::set_clear_color` 写入.
	/// `Scene::frame` 可以在调用 `acquire_frame` 之前修改该字段, 实现
	/// 逐帧变化的背景色. 默认 `DEFAULT_SURFACE_CLEAR_COLOR` (不透明黑).
	///
	/// 手上只有 sRGB 显示值时, 用 `srgb_to_linear` 换算后再写入.
	pub clear_color: [f32; 4],
}

impl Gfx {
	pub fn new(window: Arc<Window>) -> Self {
		let instance = wgpu::Instance::new(
			wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
		);

		let surface = instance
			.create_surface(window.clone())
			.expect("failed to create surface");

		let adapter = pollster::block_on(instance.request_adapter(
			&wgpu::RequestAdapterOptions {
				power_preference: wgpu::PowerPreference::HighPerformance,
				compatible_surface: Some(&surface),
				force_fallback_adapter: false,
				apply_limit_buckets: false,
			},
		))
			.expect("no suitable adapter");

		let (device, queue) =
			pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
				label: Some("example_device"),
				required_features: wgpu::Features::empty(),
				required_limits: wgpu::Limits::default(),
				..Default::default()
			}))
				.expect("failed to create device");

		let device = Arc::new(device);
		let queue = Arc::new(queue);

		let size = window.inner_size();
		let width = size.width.max(1);
		let height = size.height.max(1);

		let mut config = configure_surface(&surface, &adapter, &device, width, height)
			.expect("failed to configure surface");
		config.present_mode = wgpu::PresentMode::Fifo;

		let window_format = ColorFormat::from_wgpu(config.format)
			.expect("surface format not supported");

		let mut system = RenderSystem::new(RenderSystemDescriptor {
			device: device.clone(),
			queue: queue.clone(),
			max_vertices: 65536,
			max_objects: 16,
		});

		let mut camera = PerspectiveCamera::new(
			60_f32.to_radians(),
			width as f32 / height as f32,
			0.1,
			100.0,
		);
		camera.position = Vec3::new(0.0, 0.0, 5.0);
		camera.look_at(Vec3::ZERO, Vec3::Y);
		camera.apply(system.globals_mut());

		let depth = DepthBuffer::new(&device, width, height, DepthFormat::Depth32Float, 4)
			.expect("failed to create depth buffer");

		let msaa = MsaaBuffer::new(&device, width, height, window_format, 4)
			.expect("failed to create msaa buffer");

		Self {
			window,
			surface,
			device,
			queue,
			config,
			window_format,
			system,
			camera,
			depth,
			msaa,
			clear_color: DEFAULT_SURFACE_CLEAR_COLOR,
		}
	}

	pub fn resize(&mut self, width: u32, height: u32) {
		let width = width.max(1);
		let height = height.max(1);

		if width == self.config.width && height == self.config.height {
			return;
		}

		self.config.width = width;
		self.config.height = height;
		self.surface.configure(&self.device, &self.config);

		self.rebuild_attachments(width, height);
	}

	/// 获取当前帧的 surface frame, 并把 `self.clear_color` 写入.
	///
	/// `SurfaceOutdated` 时**强制** `surface.configure`, 然后重试一次.
	/// 应用侧只需修改 `gfx.clear_color`, 不需要自己处理交换链重设.
	pub fn acquire_frame(&mut self) -> Option<SurfaceFrame> {
		for _ in 0..2 {
			match SurfaceFrame::acquire(
				&self.surface,
				&self.config,
				Some(&self.depth),
				Some(&self.msaa),
			) {
				Ok(mut f) => {
					f.set_clear_color(self.clear_color);
					return Some(f);
				}
				Err(PipelineError::SurfaceOutdated) => {
					let size = self.window.inner_size();
					let w = size.width.max(1);
					let h = size.height.max(1);
					let size_changed = w != self.config.width || h != self.config.height;

					self.config.width = w;
					self.config.height = h;
					self.surface.configure(&self.device, &self.config);

					if size_changed {
						self.rebuild_attachments(w, h);
					}
				}
				Err(PipelineError::SurfaceTimeout | PipelineError::SurfaceOccluded) => {
					return None;
				}
				Err(e) => {
					eprintln!("surface acquire error: {e:?}");
					return None;
				}
			}
		}
		None
	}

	pub fn request_redraw(&self) {
		self.window.request_redraw();
	}

	fn rebuild_attachments(&mut self, width: u32, height: u32) {
		self.depth = DepthBuffer::new(
			&self.device,
			width,
			height,
			DepthFormat::Depth32Float,
			4,
		)
			.expect("failed to resize depth buffer");

		self.msaa = MsaaBuffer::new(
			&self.device,
			width,
			height,
			self.window_format,
			4,
		)
			.expect("failed to resize msaa buffer");

		self.camera.set_viewport(width, height);
		self.camera.apply(self.system.globals_mut());
	}
}

// ================ Scene 与 Runner ================

pub trait Scene: Sized {
	fn new(gfx: &Gfx) -> Self;
	fn frame(&mut self, gfx: &mut Gfx, dt: f32, time: f32);
}

pub struct Runner<S: Scene> {
	gfx: Option<Gfx>,
	scene: Option<S>,
	last_frame: Instant,
	time: f32,
}

impl<S: Scene> Runner<S> {
	pub fn new() -> Self {
		Self {
			gfx: None,
			scene: None,
			last_frame: Instant::now(),
			time: 0.0,
		}
	}
}

impl<S: Scene + 'static> ApplicationHandler for Runner<S> {
	fn resumed(&mut self, event_loop: &ActiveEventLoop) {
		if self.scene.is_some() {
			return;
		}

		let window = Arc::new(
			event_loop
				.create_window(
					Window::default_attributes()
						.with_title("blaze3d example")
						.with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0)),
				)
				.expect("failed to create window"),
		);

		let gfx = Gfx::new(window);
		let scene = S::new(&gfx);

		self.gfx = Some(gfx);
		self.scene = Some(scene);
		self.last_frame = Instant::now();
		self.time = 0.0;
	}

	fn window_event(
		&mut self,
		event_loop: &ActiveEventLoop,
		_id: WindowId,
		event: WindowEvent,
	) {
		if matches!(event, WindowEvent::RedrawRequested) {
			let now = Instant::now();
			let dt = (now - self.last_frame).as_secs_f32();
			self.last_frame = now;
			self.time += dt;
			let time = self.time;

			let (Some(gfx), Some(scene)) = (self.gfx.as_mut(), self.scene.as_mut()) else {
				return;
			};
			scene.frame(gfx, dt, time);
			return;
		}

		match event {
			WindowEvent::CloseRequested => event_loop.exit(),

			WindowEvent::Resized(size) => {
				if let Some(gfx) = self.gfx.as_mut() {
					gfx.resize(size.width, size.height);
				}
			}

			WindowEvent::KeyboardInput {
				event:
				KeyEvent {
					logical_key: Key::Named(NamedKey::Escape),
					state: ElementState::Pressed,
					..
				},
				..
			} => event_loop.exit(),

			_ => {}
		}
	}

	fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
		if let Some(gfx) = self.gfx.as_ref() {
			gfx.request_redraw();
		}
	}
}