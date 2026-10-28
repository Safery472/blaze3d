// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 彩色立方体 —— 单 pass, 直接渲染到窗口, 带 4x MSAA 与垂直同步.
//!
//! # 编译时配置
//!
//! ```bash
//! CUBE3D_Y_SPEED=1.2 CUBE3D_X_SPEED=0.4 \
//! CUBE3D_BG_R=0.2 CUBE3D_BG_G=0.2 CUBE3D_BG_B=0.2 \
//! cargo build --release --example cube3d
//! ```
//!
//! 按 `Esc` 退出.

// wgpu 的类型图较深 (ShaderModule -> DispatchShaderModule -> Arc<CoreShaderModule>
// -> ... -> ComputePass). 证明这些类型满足 Send 时, trait 求解的嵌套深度超过
// 默认上限 128. 提高上限以容纳 wgpu 的类型图; 只影响编译期, 不影响运行时.
#![recursion_limit = "256"]

mod common;

use std::sync::Arc;

use common::{read_f32, srgb_to_linear, Gfx, Runner, Scene};
use glam::Mat4;

use blaze3d::prelude::*;
use blaze3d::vertex::{formats, BufferBuilder, DrawMode, VertexConsumer};

// ================ 本 example 的配置 ================

/// `cube3d` 自己的编译时配置.
mod config {
	use super::{read_f32, srgb_to_linear};

	/// Bevy 默认背景色的 sRGB 值, 三个通道相同.
	const BEVY_BG_SRGB: f32 = 0.11;

	/// 横向 (绕 Y 轴) 角速度, 弧度/秒. 默认 0.6.
	pub fn y_speed() -> f32 {
		read_f32(option_env!("CUBE3D_Y_SPEED"), 0.6)
	}

	/// 纵向 (绕 X 轴) 角速度, 弧度/秒. 默认 0.2.
	pub fn x_speed() -> f32 {
		read_f32(option_env!("CUBE3D_X_SPEED"), 0.2)
	}

	/// 背景色, RGBA **线性空间**, 默认 Bevy 的 sRGB `(0.11, 0.11, 0.11)`.
	///
	/// 环境变量以 sRGB 值给出, 函数内部换算成线性, 直接传给
	/// `SurfaceFrame::set_clear_color` 后屏幕显示的就是给定 sRGB 值.
	pub fn bg_color() -> [f32; 4] {
		let r = read_f32(option_env!("CUBE3D_BG_R"), BEVY_BG_SRGB);
		let g = read_f32(option_env!("CUBE3D_BG_G"), BEVY_BG_SRGB);
		let b = read_f32(option_env!("CUBE3D_BG_B"), BEVY_BG_SRGB);
		[
			srgb_to_linear(r),
			srgb_to_linear(g),
			srgb_to_linear(b),
			1.0,
		]
	}
}

// ================ 场景几何 ================

fn create_cube(
	device: &wgpu::Device,
	queue: &wgpu::Queue,
) -> Result<GpuMesh, blaze3d::system::Error> {
	const RED: [u8; 4] = [255, 100, 100, 255];
	const GREEN: [u8; 4] = [100, 255, 100, 255];
	const BLUE: [u8; 4] = [100, 100, 255, 255];
	const YELLOW: [u8; 4] = [255, 255, 100, 255];
	const MAGENTA: [u8; 4] = [255, 100, 255, 255];
	const CYAN: [u8; 4] = [100, 255, 255, 255];

	const FACES: [([[f32; 3]; 4], [u8; 4]); 6] = [
		([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]], RED),
		([[1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0]], GREEN),
		([[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]], BLUE),
		([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]], YELLOW),
		([[1.0, -1.0, 1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]], MAGENTA),
		([[-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]], CYAN),
	];

	let mut builder = BufferBuilder::with_capacity(24 * 16);
	let mut b = builder.begin(DrawMode::Quads, formats::position_color());

	for (positions, color) in FACES {
		for pos in positions {
			b.vertex(pos[0], pos[1], pos[2]).expect("vertex");
			b.color(color[0], color[1], color[2], color[3]).expect("color");
			b.end_vertex().expect("end_vertex");
		}
	}

	let rendered = b.end();
	GpuMesh::new(device, queue, &rendered)
}

// ================ Scene 实现 ================

struct Cube3d {
	render_type: Arc<RenderType>,
	shader: Shader,
	layout: LayoutHandle,
	mesh: GpuMesh,
	angle_x: f32,
	angle_y: f32,
	y_speed: f32,
	x_speed: f32,
	bg_color: [f32; 4],
}

impl Scene for Cube3d {
	fn new(gfx: &Gfx) -> Self {
		let mesh = create_cube(&gfx.device, &gfx.queue).expect("failed to create cube");
		let shader = BuiltinShader::PositionColor.create(&gfx.device);

		let render_type = Arc::new(
			RenderType::builder("cube")
				.vertex_format(formats::position_color())
				.color_format(gfx.window_format)
				.depth(DepthMode::read_write())
				.depth_format(Some(DepthFormat::Depth32Float))
				.cull(CullMode::Back)
				.sample_count(4)
				.build(),
		);

		let globals_layout = gfx.system.globals_bind_group_layout().clone();
		let object_layout = gfx.system.object_bind_group_layout().clone();
		let layout = LayoutHandle::new(&gfx.device, &[&globals_layout, &object_layout]);

		Self {
			render_type,
			shader,
			layout,
			mesh,
			angle_x: 0.0,
			angle_y: 0.0,
			y_speed: config::y_speed(),
			x_speed: config::x_speed(),
			bg_color: config::bg_color(),
		}
	}

	fn frame(&mut self, gfx: &mut Gfx, dt: f32, _time: f32) {
		self.angle_y += dt * self.y_speed;
		self.angle_x += dt * self.x_speed;

		// 写回背景色, `acquire_frame` 内部会把它传给
		// `SurfaceFrame::set_clear_color`.
		gfx.clear_color = self.bg_color;

		let model =
			Mat4::from_rotation_x(self.angle_x) * Mat4::from_rotation_y(-self.angle_y);

		let Some(surface_frame) = gfx.acquire_frame() else {
			return;
		};

		let offset = gfx.system.push_object(model).expect("object buffer full");

		let mut frame = gfx.system.begin_frame();
		frame.pass(&surface_frame, |pass| {
			pass.set_render_type(&self.render_type, &self.shader, &self.layout);
			pass.set_object(offset);
			pass.draw_mesh(&self.mesh);
		});
		frame.submit();

		surface_frame.present(&gfx.queue);
	}
}

fn main() {
	let event_loop = winit::event_loop::EventLoop::new().expect("failed to create event loop");
	event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

	let mut runner = Runner::<Cube3d>::new();
	event_loop.run_app(&mut runner).expect("event loop failed");
}