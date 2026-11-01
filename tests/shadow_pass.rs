// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 阴影 pass 的端到端测试.
//!
//! 覆盖两件事:
//!
//! 1. `CubeShadowMap::render_face` 能直接交给 `RenderSystem::render_frame`,
//!    并且生成的 render pass **没有颜色附件**.
//! 2. 三个内置阴影着色器都能与对应的绘制方法配上, 把几何写进深度附件.
//!
//! # 设备
//!
//! 拿不到设备时静默跳过 —— 无 GPU 的 CI 不应因此失败.

// wgpu 的类型图较深, 证明 `Send` 时会超出默认的 trait 求解深度.
#![recursion_limit = "256"]

use std::sync::Arc;

use blaze3d::pipeline::{RenderTarget, Target};
use blaze3d::pipeline::{CubeFace, CubeShadowMap};
use blaze3d::prelude::*;
use blaze3d::state::{ColorFormat, DepthFormat};
use blaze3d::system::{RenderSystem, RenderSystemDescriptor};
use blaze3d::vertex::{formats, BufferBuilder, DrawMode, VertexConsumer};

/// 取一个可用设备.
fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
	let instance =
		wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
	let request = |fallback: bool| {
		pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
			power_preference: wgpu::PowerPreference::LowPower,
			compatible_surface: None,
			force_fallback_adapter: fallback,
			apply_limit_buckets: false,
		}))
		.ok()
	};
	let adapter = request(true).or_else(|| request(false))?;
	let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
		label: Some("shadow_tests"),
		required_features: wgpu::Features::empty(),
		required_limits: wgpu::Limits::default(),
		..Default::default()
	}))
	.ok()?;
	Some((device, queue))
}

/// 一个覆盖 `[-0.5, 0.5]^3` 的三角形, 只带位置.
fn position_triangle(device: &wgpu::Device, queue: &wgpu::Queue) -> GpuMesh {
	let mut builder = BufferBuilder::with_capacity(64);
	let mut b = builder.begin(DrawMode::Triangles, formats::position_only());
	for (x, y, z) in [
		(-0.5, -0.5, 0.0),
		(0.5, -0.5, 0.0),
		(0.0, 0.5, 0.0),
	] {
		b.vertex(x, y, z).expect("vertex");
		b.end_vertex().expect("end vertex");
	}
	let rendered = b.end();
	GpuMesh::new(device, queue, &rendered).expect("mesh")
}

/// 带骨骼属性的三角形.
fn skinned_triangle(device: &wgpu::Device, queue: &wgpu::Queue) -> GpuMesh {
	let mut builder = BufferBuilder::with_capacity(128);
	let mut b = builder.begin(DrawMode::Triangles, formats::position_color_skinned());
	for (x, y, z) in [
		(-0.5, -0.5, 0.0),
		(0.5, -0.5, 0.0),
		(0.0, 0.5, 0.0),
	] {
		b.vertex(x, y, z).expect("vertex");
		b.color(255, 255, 255, 255).expect("color");
		b.bone_index([0, 0, 0, 0]).expect("bone index");
		b.bone_weight([255, 0, 0, 0]).expect("bone weight");
		b.end_vertex().expect("end vertex");
	}
	let rendered = b.end();
	GpuMesh::new(device, queue, &rendered).expect("mesh")
}

fn system(device: &wgpu::Device, queue: &wgpu::Queue) -> RenderSystem {
	RenderSystem::new(RenderSystemDescriptor {
		device: Arc::new(device.clone()),
		queue: Arc::new(queue.clone()),
		max_vertices: 256,
		max_objects: 8,
	})
}

/// 立方体阴影贴图的单面应当可以直接作为 `render_frame` 的目标.
#[test]
fn cube_face_target_renders_without_color_attachment() {
	let Some((device, queue)) = device() else {
		eprintln!("no GPU adapter; skipping");
		return;
	};

	let mut shadow = CubeShadowMap::new(&device, 64, glam::Vec3::new(0.0, 0.0, 2.0), 32.0)
		.expect("cube shadow");
	let mesh = position_triangle(&device, &queue);
	let mut system = system(&device, &queue);

	let shader = BuiltinShader::ShadowDepthMesh.create(&device);
	let globals_layout = system.globals_bind_group_layout().clone();
	let object_layout = system.object_bind_group_layout().clone();
	let layout = LayoutHandle::new(&device, &[&globals_layout, &object_layout]);

	let render_type = Arc::new(
		RenderType::builder("shadow_face")
			.vertex_format(formats::position_only())
			.depth_only()
			.depth_format(Some(DepthFormat::Depth32Float))
			.write_mask(WriteMask::DEPTH_ONLY)
			.build(),
	);

	let offset = system.push_object(glam::Mat4::IDENTITY).expect("object");

	for face in CubeFace::ALL {
		system.globals_mut().view_projection =
			shadow.light_view_projection(face).to_cols_array_2d();

		let face_target = shadow.render_face(face);
		// 无颜色附件是这套 API 的关键点: 阴影 pass 不该消耗颜色带宽.
		assert!(
			face_target.color_attachment().is_none(),
			"{face:?}: cube shadow face must not have a colour attachment"
		);
		assert_eq!(face_target.viewport_size(), (64, 64));
		assert!(face_target.depth_attachment().is_some());

		system.render_frame(&face_target, |pass| {
			pass.set_render_type(&render_type, &shader, &layout);
			pass.set_object(offset);
			pass.draw_mesh(&mesh);
		});
	}
}

/// 普通网格阴影着色器与 `draw_mesh` 配合, 能写进深度附件.
#[test]
fn shadow_depth_mesh_shader_draws_into_a_depth_target() {
	let Some((device, queue)) = device() else {
		return;
	};

	let target = RenderTarget::new(
		&device,
		32,
		32,
		ColorFormat::Rgba8Unorm,
		Some(DepthFormat::Depth32Float),
		Some("shadow_target"),
	)
	.expect("target");

	let mut system = system(&device, &queue);
	let mesh = position_triangle(&device, &queue);
	let shader = BuiltinShader::ShadowDepthMesh.create(&device);
	let layout = LayoutHandle::new(&device, &[
		&system.globals_bind_group_layout().clone(),
		&system.object_bind_group_layout().clone(),
	]);
	let render_type = Arc::new(
		RenderType::builder("shadow_mesh")
			.vertex_format(formats::position_only())
			// `RenderTarget` 带颜色附件, 因此这里声明一个颜色目标 (只关了
			// 颜色写入). 只有 `CubeFaceTarget` 那种无颜色附件的 pass 才能
			// 用 `depth_only()`.
			.color_format(ColorFormat::Rgba8Unorm)
			.depth_format(Some(DepthFormat::Depth32Float))
			.write_mask(WriteMask::DEPTH_ONLY)
			.build(),
	);
	let offset = system.push_object(glam::Mat4::IDENTITY).expect("object");

	system.render_frame(&target, |pass| {
		pass.set_render_type(&render_type, &shader, &layout);
		pass.set_object(offset);
		pass.draw_mesh(&mesh);
	});
}

/// 蒙皮阴影着色器: 骨骼矩阵来自 group 2, 需要能把骨骼缓冲接上.
#[test]
fn shadow_depth_skinned_shader_binds_bones() {
	let Some((device, queue)) = device() else {
		return;
	};

	let target = RenderTarget::new(
		&device,
		32,
		32,
		ColorFormat::Rgba8Unorm,
		Some(DepthFormat::Depth32Float),
		Some("shadow_target_skinned"),
	)
	.expect("target");

	let mut system = system(&device, &queue);
	let mesh = skinned_triangle(&device, &queue);
	let mut bones = BoneBuffer::new(&device, 8);
	let bone_offset = bones
		.push_skeleton(&queue, &[glam::Mat4::IDENTITY])
		.expect("bones");

	let shader = BuiltinShader::ShadowDepthSkinned.create(&device);
	let globals_layout = system.globals_bind_group_layout().clone();
	let object_layout = system.object_bind_group_layout().clone();
	let layout = LayoutHandle::new(&device, &[
		&globals_layout,
		&object_layout,
		bones.bind_group_layout(),
	]);
	let render_type = Arc::new(
		RenderType::builder("shadow_skinned")
			.vertex_format(formats::position_color_skinned())
			.color_format(ColorFormat::Rgba8Unorm)
			.depth_format(Some(DepthFormat::Depth32Float))
			.write_mask(WriteMask::DEPTH_ONLY)
			.build(),
	);

	// 骨骼范围随 `model` 一起写进对象 uniform (group 1).
	let offset = system
		.push_object_skinned(glam::Mat4::IDENTITY, bone_offset, 1)
		.expect("object");

	system.render_frame(&target, |pass| {
		pass.set_render_type(&render_type, &shader, &layout);
		pass.set_object(offset);
		pass.set_bind_group(2, bones.bind_group(), &[]);
		pass.draw_mesh(&mesh);
	});
}
