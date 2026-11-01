// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 彩虹空心立方体 —— 12 条圆柱边 + 8 个角球, 边内线性渐变,
//! 整体颜色随时间流动, 透明度闪烁, 横向旋转, 4x MSAA 与垂直同步.
//!
//! # 几何
//!
//! 每条边用 8 边圆柱绘制, 两端各缩短 `radius`, 端点正好落在角球表面.
//! 每个角放一个 8×4 UV 球, 半径与圆柱一致. 圆柱端面不生成, 被球精确
//! 覆盖, 几何上没有重叠, 半透明 alpha 不会累加变深.
//!
//! # 颜色
//!
//! 每条边的起点和终点各有一个 hue, 终点比起点偏移 `EDGE_HUE_SPAN`.
//! 圆柱四边形的起点端顶点用起点色, 终点端顶点用终点色, 内部由 GPU
//! 自动插值, 形成边内线性渐变. 两个 hue 都随时间整体偏移, 所以渐变
//! 方向固定, 颜色沿边流动.
//!
//! # 编译时配置
//!
//! ```bash
//! RAINBOW_Y_SPEED=1.2 RAINBOW_X_SPEED=0.4 \
//! RAINBOW_EDGE_WIDTH=0.02 RAINBOW_FLICKER_DEPTH=0.5 RAINBOW_FLICKER_SPEED=2.0 \
//! RAINBOW_BG_R=0.05 RAINBOW_BG_G=0.05 RAINBOW_BG_B=0.05 \
//! cargo build --release --example rainbow_cube
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
use glam::{Mat4, Vec3};

use blaze3d::prelude::*;
use blaze3d::vertex::{formats, BufferBuilder, DrawMode, VertexConsumer};

// ================ 本 example 的配置 ================

/// `rainbow_cube` 自己的编译时配置.
///
/// 环境变量前缀用 `RAINBOW_`, 与 `cube3d` 的 `CUBE3D_` 区分. 每个值
/// 只在 `Scene::new` 里解析一次, 存进字段, `frame` 每帧直接读.
mod config {
	use super::{read_f32, srgb_to_linear};

	/// Bevy 默认背景色的 sRGB 值, 三个通道相同.
	const BEVY_BG_SRGB: f32 = 0.11;

	/// 横向 (绕 Y 轴) 角速度, 弧度/秒. 默认 0.6.
	pub fn y_speed() -> f32 {
		read_f32(option_env!("RAINBOW_Y_SPEED"), 0.6)
	}

	/// 纵向 (绕 X 轴) 角速度, 弧度/秒. 默认 0.2.
	pub fn x_speed() -> f32 {
		read_f32(option_env!("RAINBOW_X_SPEED"), 0.2)
	}

	/// 线条半径. 默认 0.015, 相对边长 2 约 0.75%.
	pub fn edge_radius() -> f32 {
		read_f32(option_env!("RAINBOW_EDGE_WIDTH"), 0.015)
	}

	/// 闪烁深度, 即 alpha 从 1.0 降到 `1.0 - depth` 的幅度. 默认 0.95.
	///
	/// 设为 0 时完全不闪烁, 设为 1 时从全透明到全不透明.
	pub fn flicker_depth() -> f32 {
		read_f32(option_env!("RAINBOW_FLICKER_DEPTH"), 0.95)
	}

	/// 闪烁速度, 弧度/秒. 默认 2.0.
	pub fn flicker_speed() -> f32 {
		read_f32(option_env!("RAINBOW_FLICKER_SPEED"), 2.0)
	}

	/// 背景色, RGBA **线性空间**, 默认 Bevy 的 sRGB `(0.11, 0.11, 0.11)`.
	///
	/// 环境变量以 sRGB 值给出, 函数内部换算成线性, 直接传给
	/// `SurfaceFrame::set_clear_color` 后屏幕显示的就是给定 sRGB 值.
	pub fn bg_color() -> [f32; 4] {
		let r = read_f32(option_env!("RAINBOW_BG_R"), BEVY_BG_SRGB);
		let g = read_f32(option_env!("RAINBOW_BG_G"), BEVY_BG_SRGB);
		let b = read_f32(option_env!("RAINBOW_BG_B"), BEVY_BG_SRGB);
		[
			srgb_to_linear(r),
			srgb_to_linear(g),
			srgb_to_linear(b),
			1.0,
		]
	}
}

// ================ 几何常量 ================

const CORNERS: [Vec3; 8] = [
	Vec3::new(-1.0, -1.0, -1.0),
	Vec3::new( 1.0, -1.0, -1.0),
	Vec3::new( 1.0,  1.0, -1.0),
	Vec3::new(-1.0,  1.0, -1.0),
	Vec3::new(-1.0, -1.0,  1.0),
	Vec3::new( 1.0, -1.0,  1.0),
	Vec3::new( 1.0,  1.0,  1.0),
	Vec3::new(-1.0,  1.0,  1.0),
];

const EDGES: [(usize, usize); 12] = [
	(0, 1), (1, 2), (2, 3), (3, 0),
	(4, 5), (5, 6), (6, 7), (7, 4),
	(0, 4), (1, 5), (2, 6), (3, 7),
];

const CYLINDER_SEGMENTS: usize = 8;
const SPHERE_LON: usize = 8;
const SPHERE_LAT: usize = 4;

/// 每条边终点相对起点的 hue 偏移, 形成边内渐变.
///
/// 0.15 约等于色环 54 度, 视觉上是一条边从一种颜色平滑过渡到相邻的
/// 另一种颜色.
const EDGE_HUE_SPAN: f32 = 0.15;

// ================ 工具函数 ================

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
	let h = h.rem_euclid(1.0) * 6.0;
	let i = h.floor() as i32;
	let f = h - i as f32;
	let p = v * (1.0 - s);
	let q = v * (1.0 - f * s);
	let t = v * (1.0 - (1.0 - f) * s);
	match i % 6 {
		0 => [v, t, p],
		1 => [q, v, p],
		2 => [p, v, t],
		3 => [p, q, v],
		4 => [t, p, v],
		_ => [v, p, q],
	}
}

/// 把 hue / 饱和度 / 明度 / alpha 转成 `[u8; 4]`.
fn hsv_color(h: f32, s: f32, v: f32, alpha: f32) -> [u8; 4] {
	let [r, g, b] = hsv_to_rgb(h, s, v);
	[
		(r * 255.0) as u8,
		(g * 255.0) as u8,
		(b * 255.0) as u8,
		(alpha * 255.0) as u8,
	]
}

fn perpendicular_basis(dir: Vec3) -> (Vec3, Vec3) {
	let reference = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
	let u = dir.cross(reference).normalize();
	let w = dir.cross(u).normalize();
	(u, w)
}

fn sphere_point(center: Vec3, radius: f32, phi: f32, theta: f32) -> Vec3 {
	center
		+ Vec3::new(
		phi.sin() * theta.cos(),
		phi.cos(),
		phi.sin() * theta.sin(),
	) * radius
}

// ================ 几何构建 ================

fn push_vertex(b: &mut impl VertexConsumer, p: Vec3, color: [u8; 4]) {
	b.vertex(p.x, p.y, p.z).expect("vertex");
	b.color(color[0], color[1], color[2], color[3])
		.expect("color");
	b.end_vertex().expect("end_vertex");
}

/// 圆柱, 两端各缩短 `radius`, 端面不生成, 由角球覆盖.
///
/// 圆柱侧面每个四边形跨长度方向的两个端点, 起点端顶点用 `color_start`,
/// 终点端顶点用 `color_end`. GPU 在四边形内部插值, 形成从起点到终点
/// 的线性渐变.
fn add_cylinder(
	b: &mut impl VertexConsumer,
	a: Vec3,
	c: Vec3,
	radius: f32,
	color_start: [u8; 4],
	color_end: [u8; 4],
) {
	let dir = (c - a).normalize();
	let (u, w) = perpendicular_basis(dir);
	let start = a + dir * radius;
	let end = c - dir * radius;

	let mut ring_start = [Vec3::ZERO; CYLINDER_SEGMENTS];
	let mut ring_end = [Vec3::ZERO; CYLINDER_SEGMENTS];
	for i in 0..CYLINDER_SEGMENTS {
		let theta = i as f32 / CYLINDER_SEGMENTS as f32 * std::f32::consts::TAU;
		let offset = (theta.cos() * u + theta.sin() * w) * radius;
		ring_start[i] = start + offset;
		ring_end[i] = end + offset;
	}

	for i in 0..CYLINDER_SEGMENTS {
		let j = (i + 1) % CYLINDER_SEGMENTS;
		push_vertex(b, ring_start[i], color_start);
		push_vertex(b, ring_end[i], color_end);
		push_vertex(b, ring_end[j], color_end);
		push_vertex(b, ring_start[j], color_start);
	}
}

/// UV 球. 极点处四边形退化成三角形, wgpu 会自动处理.
fn add_sphere(
	b: &mut impl VertexConsumer,
	center: Vec3,
	radius: f32,
	color: [u8; 4],
) {
	for lat in 0..SPHERE_LAT {
		let phi0 = lat as f32 / SPHERE_LAT as f32 * std::f32::consts::PI;
		let phi1 = (lat + 1) as f32 / SPHERE_LAT as f32 * std::f32::consts::PI;

		for lon in 0..SPHERE_LON {
			let theta0 = lon as f32 / SPHERE_LON as f32 * std::f32::consts::TAU;
			let theta1 = (lon + 1) as f32 / SPHERE_LON as f32 * std::f32::consts::TAU;

			let p00 = sphere_point(center, radius, phi0, theta0);
			let p01 = sphere_point(center, radius, phi0, theta1);
			let p10 = sphere_point(center, radius, phi1, theta0);
			let p11 = sphere_point(center, radius, phi1, theta1);

			for p in [p00, p10, p11, p01] {
				push_vertex(b, p, color);
			}
		}
	}
}

/// 按当前时间构建整个空心立方体: 12 条渐变圆柱 + 8 个角球.
fn build_rainbow_cube(
	device: &wgpu::Device,
	queue: &wgpu::Queue,
	time: f32,
	radius: f32,
	flicker_depth: f32,
	flicker_speed: f32,
) -> UploadedBuffer {
	// `abs(sin)` 让谷值变成尖峰, alpha 快速穿过最低点, 不会长时间停在
	// 低值附近. 与 `0.5 + 0.5 * sin` 相比:
	//
	// - 峰值仍然是 1.0, 谷值是 `1.0 - flicker_depth`.
	// - 周期减半, 两次闪烁间隔缩短, 观感更"闪"而不是"呼吸".
	// - 谷值附近导数不为零 (尖点), alpha 不会在低值停留.
	let alpha = 1.0 - flicker_depth * (time * flicker_speed).sin().abs();

	let capacity = (EDGES.len() * CYLINDER_SEGMENTS + CORNERS.len() * SPHERE_LON * SPHERE_LAT)
		* 4
		* 16;

	let mut builder = BufferBuilder::with_capacity(capacity);
	let mut b = builder.begin(DrawMode::Quads, formats::position_color());

	// 圆柱: 每条边有起点和终点两个 hue.
	for (i, &(ai, bi)) in EDGES.iter().enumerate() {
		let hue_start = i as f32 / EDGES.len() as f32 + time * 0.2;
		let hue_end = hue_start + EDGE_HUE_SPAN;

		let color_start = hsv_color(hue_start, 0.9, 1.0, alpha);
		let color_end = hsv_color(hue_end, 0.9, 1.0, alpha);

		add_cylinder(&mut b, CORNERS[ai], CORNERS[bi], radius, color_start, color_end);
	}

	// 角球: 独立 hue, 也随时间偏移.
	for (i, &corner) in CORNERS.iter().enumerate() {
		let hue = i as f32 / CORNERS.len() as f32 + time * 0.2;
		let color = hsv_color(hue, 0.9, 1.0, alpha);
		add_sphere(&mut b, corner, radius, color);
	}

	let rendered = b.end();
	rendered.upload(device, queue)
}

// ================ Scene 实现 ================

struct RainbowCube {
	render_type: Arc<RenderType>,
	shader: Shader,
	layout: LayoutHandle,
	angle_x: f32,
	angle_y: f32,
	y_speed: f32,
	x_speed: f32,
	radius: f32,
	flicker_depth: f32,
	flicker_speed: f32,
	bg_color: [f32; 4],
}

impl Scene for RainbowCube {
	fn new(gfx: &Gfx) -> Self {
		let shader = BuiltinShader::PositionColor.create(&gfx.device);

		let render_type = Arc::new(
			RenderType::builder("rainbow_cube")
				.vertex_format(formats::position_color())
				.color_format(gfx.window_format)
				.depth(DepthMode::read_write())
				.depth_format(Some(DepthFormat::Depth32Float))
				.cull(CullMode::None)
				.blend(BlendMode::Alpha)
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
			angle_x: 0.0,
			angle_y: 0.0,
			y_speed: config::y_speed(),
			x_speed: config::x_speed(),
			radius: config::edge_radius(),
			flicker_depth: config::flicker_depth(),
			flicker_speed: config::flicker_speed(),
			bg_color: config::bg_color(),
		}
	}

	fn frame(&mut self, gfx: &mut Gfx, dt: f32, time: f32) {
		self.angle_y += dt * self.y_speed;
		self.angle_x += dt * self.x_speed;

		// 写回背景色, `acquire_frame` 内部会把它传给
		// `SurfaceFrame::set_clear_color`.
		gfx.clear_color = self.bg_color;

		let model =
			Mat4::from_rotation_x(self.angle_x) * Mat4::from_rotation_y(-self.angle_y);

		let uploaded = build_rainbow_cube(
			&gfx.device,
			&gfx.queue,
			time,
			self.radius,
			self.flicker_depth,
			self.flicker_speed,
		);

		let Some(surface_frame) = gfx.acquire_frame() else {
			return;
		};

		let offset = gfx.system.push_object(model).expect("object buffer full");

		let mut frame = gfx.system.begin_frame();
		frame.pass(&surface_frame, |pass| {
			pass.set_render_type(&self.render_type, &self.shader, &self.layout);
			pass.set_object(offset);
			pass.draw_uploaded(&uploaded);
		});
		frame.submit();

		surface_frame.present(&gfx.queue);
	}
}

fn main() {
	let event_loop = winit::event_loop::EventLoop::new().expect("failed to create event loop");
	event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

	let mut runner = Runner::<RainbowCube>::new();
	event_loop.run_app(&mut runner).expect("event loop failed");
}