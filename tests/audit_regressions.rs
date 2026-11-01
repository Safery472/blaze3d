// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! CPU 回归测试.
//!
//! 只覆盖不依赖 GPU 的路径: 构造校验, 采样逻辑与纯数学投影. 需要设备的
//! 用例 (纹理上传, mipmap 生成, 管线创建) 不在本模块 —— 它们在无图形
//! 环境的 CI 上无法运行, 需要时挂 `#[ignore]` 单独跑.
//!
//! 每个用例对应一次审计里确认过的缺陷, 用例名写明被固定的行为, 避免
//! 后人把修复改回去.

use glam::Vec3;

use blaze3d::camera::Aabb;
use blaze3d::pipeline::ShadowMap;
use blaze3d::system::{BoneTrack, BoneTransform};
use blaze3d::vertex::element::{ElementType, ElementUsage};
use blaze3d::vertex::{
	Direction, VertexFormat, VertexFormatElement, error::Error as VertexError,
};

// ================ 辅助 ================

/// 构造一条长度为 `n` 的恒等变换序列.
fn identities(n: usize) -> Vec<BoneTransform> {
	vec![BoneTransform::identity(); n]
}

// ================ 骨骼动画 ================

/// NaN 时间比较恒为 `false`, 只靠严格升序检查会放行.
fn assert_invalid_time(times: Vec<f32>) {
	let n = times.len();
	let result = BoneTrack::new(0, times, identities(n));
	assert!(
		matches!(result, Err(blaze3d::system::Error::TrackInvalidTime { bone: 0 })),
		"non-finite or negative keyframe time must be rejected"
	);
}

#[test]
fn bone_track_rejects_nan_time() {
	assert_invalid_time(vec![f32::NAN, 1.0]);
	assert_invalid_time(vec![0.0, f32::NAN]);
}

#[test]
fn bone_track_rejects_infinite_and_negative_time() {
	assert_invalid_time(vec![f32::INFINITY]);
	assert_invalid_time(vec![f32::NEG_INFINITY, 1.0]);
	assert_invalid_time(vec![-0.5, 1.0]);
}

#[test]
fn bone_track_sample_clamps_at_the_ends() {
	let first = BoneTransform {
		translation: Vec3::X,
		..BoneTransform::identity()
	};
	let last = BoneTransform {
		translation: Vec3::Y,
		..BoneTransform::identity()
	};
	let track = BoneTrack::new(3, vec![0.0, 2.0], vec![first, last]).expect("track");

	assert_eq!(track.sample(-1.0), Some(first));
	assert_eq!(track.sample(0.0), Some(first));
	assert_eq!(track.sample(3.0), Some(last));

	// 中点: 取平移的线性插值, 即各轴各占一半.
	let mid = track.sample(1.0).expect("midpoint");
	assert!((mid.translation - Vec3::new(0.5, 0.5, 0.0)).length() < 1e-6);
}

/// `sample` 不能因为绕过 `new` 的畸形数据而 panic.
#[test]
fn bone_track_sample_survives_malformed_track() {
	let track = BoneTrack {
		bone: 0,
		times: vec![f32::NAN, 1.0],
		transforms: identities(2),
	};
	// 下溢路径: partition_point 返回 0.
	assert!(track.sample(0.5).is_some());
	assert!(track.sample(0.0).is_some());
}

// ================ 顶点格式 ================

#[test]
fn normal_byte_x3_occupies_four_byte_slot() {
	let element = VertexFormatElement::normal_i8x3();
	assert_eq!(element.byte_size(), 4);
	assert_eq!(element.count(), 3);
}

#[test]
fn normal_only_format_stride_is_aligned() {
	// 3 字节会撞上 wgpu 的 VERTEX_ALIGNMENT; 4 字节槽位让 stride 合法.
	let format =
		VertexFormat::new([("Normal", VertexFormatElement::normal_i8x3())]).expect("format");
	assert_eq!(format.vertex_size(), 4);
	assert_eq!(format.vertex_size() % 4, 0);

	let layout = format.wgpu_layout();
	assert_eq!(layout.array_stride, 4);
	assert_eq!(layout.attributes.len(), 1);
}

#[test]
fn preset_formats_keep_their_stride() {
	let cases = [
		(blaze3d::vertex::formats::position_only(), 12),
		(blaze3d::vertex::formats::position_color(), 16),
		(blaze3d::vertex::formats::position_tex(), 20),
		(blaze3d::vertex::formats::position_color_tex(), 24),
		(blaze3d::vertex::formats::position_color_lightmap(), 20),
		// Normal 占 4 字节槽位: 12 + 4 + 8 + 4 + 4 = 32, 加末尾 4 字节
		// `Padding` 后 stride 为 36; new_entity 多一个 UV1 (4 字节), 为 40.
		(blaze3d::vertex::formats::block(), 36),
		(blaze3d::vertex::formats::new_entity(), 40),
		(blaze3d::vertex::formats::position_color_skinned(), 24),
	];
	for (format, expected) in cases {
		assert_eq!(format.vertex_size(), expected, "unexpected stride for {format:?}");
		assert_eq!(format.vertex_size() % 4, 0);
	}
}

#[test]
fn oversized_padding_returns_stride_overflow() {
	let padding = VertexFormatElement::new(
		0,
		ElementType::Float,
		ElementUsage::Padding,
		u32::MAX,
	)
	.expect("element");
	assert_eq!(padding.checked_byte_size(), None);

	let result = VertexFormat::new([
		("Position", VertexFormatElement::position_f32x3()),
		("Padding", padding),
	]);
	assert!(
		matches!(result, Err(VertexError::StrideOverflow)),
		"oversized element must return StrideOverflow instead of panicking"
	);
}

#[test]
fn misaligned_stride_is_rejected() {
	// Position 12 + UByte x1 填充 1 = 13, 不是 4 的倍数. 两个元素都能
	// 映射到 wgpu 格式, 因此失败原因必然是对齐.
	let pad = VertexFormatElement::new(0, ElementType::UByte, ElementUsage::Padding, 1)
		.expect("padding");

	let result = VertexFormat::new([
		("Position", VertexFormatElement::position_f32x3()),
		("Pad", pad),
	]);

	match result {
		Err(VertexError::StrideMisaligned { stride }) => assert_eq!(stride, 13),
		other => panic!("expected StrideMisaligned, got {other:?}"),
	}
}

#[test]
fn block_preset_layout_matches_docs() {
	// 文档里的偏移表: 改 `formats::block` 的布局时必须同步这里.
	let format = blaze3d::vertex::formats::block();
	let expected = [
		("Position", 0u32, 12u32),
		("Color", 12, 4),
		("UV0", 16, 8),
		("UV2", 24, 4),
		("Normal", 28, 4),
		("Padding", 32, 4),
	];
	assert_eq!(format.vertex_size(), 36);

	for (i, (name, offset, size)) in expected.into_iter().enumerate() {
		assert_eq!(format.element_names()[i], name);
		assert_eq!(format.offset_of(i), Some(offset), "offset of {name}");
		assert_eq!(format.elements()[i].byte_size(), size, "size of {name}");
	}
}

#[test]
fn entity_preset_layout_matches_docs() {
	let format = blaze3d::vertex::formats::new_entity();
	let expected = [
		("Position", 0u32, 12u32),
		("Color", 12, 4),
		("UV0", 16, 8),
		("UV1", 24, 4),
		("UV2", 28, 4),
		("Normal", 32, 4),
		("Padding", 36, 4),
	];
	assert_eq!(format.vertex_size(), 40);

	for (i, (name, offset, size)) in expected.into_iter().enumerate() {
		assert_eq!(format.element_names()[i], name);
		assert_eq!(format.offset_of(i), Some(offset), "offset of {name}");
		assert_eq!(format.elements()[i].byte_size(), size, "size of {name}");
	}
}

// ================ 朝向 ================

#[test]
fn direction_nearest_zero_vector_is_up() {
	assert_eq!(Direction::nearest(Vec3::ZERO), Direction::Up);
}

#[test]
fn direction_nearest_axes() {
	assert_eq!(Direction::nearest(Vec3::Y), Direction::Up);
	assert_eq!(Direction::nearest(Vec3::NEG_Y), Direction::Down);
	assert_eq!(Direction::nearest(Vec3::X), Direction::East);
	assert_eq!(Direction::nearest(Vec3::NEG_X), Direction::West);
	assert_eq!(Direction::nearest(Vec3::Z), Direction::South);
	assert_eq!(Direction::nearest(Vec3::NEG_Z), Direction::North);
}

// ================ 阴影投影 ================
//
// `ShadowMap::new` 需要设备, 但矩阵本身是纯数学 —— 把算法拆成不碰 GPU
// 的公开入口后, 这里就能直接断言"任意半径的场景中心都落在光源视锥
// 内". 审计时的实现取 `near = 0.1`, `far = 4 * radius`, 半径小于
// 0.025 的场景深度会越界, 本用例因此失败.

#[test]
fn shadow_projection_covers_small_scenes() {
	for radius in [0.0_f32, 0.005, 0.01, 0.02, 0.025, 0.05, 0.5, 8.0] {
		let bounds = Aabb::new(Vec3::splat(-radius), Vec3::splat(radius));
		let vp = ShadowMap::light_view_projection_for(bounds, Vec3::NEG_Y);

		let center = bounds.center().extend(1.0);
		let clip = vp * center;
		let ndc = clip / clip.w;

		assert!(
			ndc.is_finite(),
			"radius {radius}: projected scene center is not finite (ndc = {ndc})"
		);
		assert!(
			(-1.0..=1.0).contains(&ndc.x) && (-1.0..=1.0).contains(&ndc.y),
			"radius {radius}: center outside the light frustum (ndc = {ndc})"
		);
		assert!(
			(0.0..=1.0).contains(&ndc.z),
			"radius {radius}: center depth outside [0, 1] (ndc.z = {})",
			ndc.z
		);
	}
}

#[test]
fn shadow_projection_rejects_zero_depth_scenes() {
	// 完全退化的包围盒不能被除零 —— 半径有 0.01 的下限.
	let bounds = Aabb::new(Vec3::ZERO, Vec3::ZERO);
	let vp = ShadowMap::light_view_projection_for(bounds, Vec3::ZERO);
	assert!(vp.is_finite(), "degenerate bounds must not produce NaN/inf");
}

// ================ 纹理 ================

#[test]
fn max_mip_levels_matches_dimensions() {
	use blaze3d::texture::image::max_mip_levels_for_test as max_mips;
	use blaze3d::texture::TextureDesc;

	let d2 = TextureDesc::new_2d(4, 2, wgpu::TextureFormat::Rgba8Unorm);
	assert_eq!(max_mips(&d2), 3);

	let cube = TextureDesc::new_cube(1, wgpu::TextureFormat::Rgba8Unorm);
	assert_eq!(max_mips(&cube), 1);

	let d3 = TextureDesc::new_3d(2, 2, 8, wgpu::TextureFormat::Rgba8Unorm);
	assert_eq!(max_mips(&d3), 4);

	// D2 的 `depth` 字段不参与上限计算.
	let d2_flat = TextureDesc::new_2d(2, 2, wgpu::TextureFormat::Rgba8Unorm);
	assert_eq!(max_mips(&d2_flat), 2);
}
