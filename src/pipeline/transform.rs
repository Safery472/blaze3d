// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 阴影模块共用的矩阵构造.
//!
//! 方向光阴影 ([`super::shadow`]) 与点光阴影 ([`super::cube`]) 都需要从
//! 光源位置与朝向构造视图矩阵. 两者之前各抄了一份实现, 行为必须保持
//! 一致 —— 否则同方向的光在两类阴影贴图里会算出不用的结果.

use glam::{Mat4, Vec3};

/// 构造右手系 look-at 视图矩阵.
///
/// `glam 0.34` 移除了 `Mat4::look_at_rh`, 因此这里手工构造. 旋转部分
/// 与 [`crate::camera::PerspectiveCamera::look_at`] 一致, 区别在于本
/// 函数直接返回完整的视图矩阵, 而不是旋转四元数.
///
/// # 参数
///
/// - `eye` —— 相机位置.
/// - `target` —— 注视点.
/// - `up` —— 世界空间的「上」方向. 与视线平行时结果未定义, 调用方
///   需自行保证不退化.
pub(crate) fn look_at_rh(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
	// `away` 即相机的 +Z 轴 (背离视线方向).
	let away = (eye - target).normalize();
	let right = up.cross(away).normalize();
	let up = away.cross(right);

	// 相机在世界空间的变换: 前三列依次是 `right` / `up` / `away`,
	// 平移列是 `eye`.
	let world = Mat4::from_cols(
		right.extend(0.0),
		up.extend(0.0),
		away.extend(0.0),
		eye.extend(1.0),
	);

	// 视图矩阵即世界变换的逆矩阵.
	world.inverse()
}
