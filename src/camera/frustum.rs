// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 视锥 (`Frustum`) 与平面 (`Plane`).
//!
//! [`Frustum`] 是由六个归一化平面构成的集合, 从视图投影矩阵中提取, 供
//! [`Aabb::intersects_frustum`] 做剔除判定.
//!
//! # 坐标系
//!
//! 提取公式依赖 **NDC 深度范围 `[0, 1]`**, 即 wgpu / Vulkan 的约定.
//! `OpenGL` 使用 `[-1, 1]`, near 平面的公式并不相同. 本模块不提供
//! `OpenGL` 版本 —— 本项目的整条链路都基于 wgpu.
//!
//! # 提取方法
//!
//! 采用 Gribb-Hartmann 方法 —— 直接从视图投影矩阵的**行**读出六个平面.
//!
//! ```text
//! 令 vp 的第 i 行为 r_i  (i = 0..3)
//!
//! left   = r3 + r0
//! right  = r3 - r0
//! bottom = r3 + r1
//! top    = r3 - r1
//! near   = r2            ← wgpu 特有: z ≥ 0
//! far    = r3 - r2
//! ```
//!
//! # 行 vs 列
//!
//! **必须按行提取**. 按列会得到完全错误的平面位置 —— 例如 near 平面会
//! 落到相机背后, 近处物体因此被误判为可见.
//!
//! [`glam::Mat4`] 采用列主序存储 —— `to_cols_array_2d()` 返回的 `m[i]`
//! 是第 i **列**, 第 i **行**则是 `(m[0][i], m[1][i], m[2][i], m[3][i])`.
//!
//! # 与相机的关系
//!
//! 从 [`crate::camera::Camera::view_projection`] 取到的矩阵可直接传给
//! [`Frustum::from_view_projection`] 使用.

use glam::{Mat4, Vec3, Vec4};

use super::aabb::Aabb;

/// 归一化的平面.
///
/// # 平面方程
///
/// `normal · p + d = 0`. `normal` 指向**视锥内侧** —— 正侧意味着
/// 「位于视锥内部」.
///
/// # 归一化
///
/// `Frustum::from_view_projection` 返回的平面均已归一化 —— `normal` 为
/// 单位向量, `d` 是原点到平面的有符号距离. 因此 `normal · p + d` 的
/// 绝对值等于点到平面的实际距离.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
	/// 平面法线, 单位向量, 指向视锥内侧.
	pub normal: Vec3,
	/// 平面方程的常数项.
	pub d: f32,
}

impl Plane {
	/// 由四个分量构造平面.
	///
	/// `(normal.xyz, d)` 对应方程 `ax + by + cz + d = 0`. 内部会做归一化,
	/// 使 `normal` 长度为 1.
	///
	/// # 退化
	///
	/// `normal` 长度接近 0 时, 归一化会产生 `NaN`. 调用方必须保证传入的
	/// 平面方程非退化; 正常的视图投影矩阵不会产生退化平面.
	fn from_components(v: Vec4) -> Self {
		let normal = Vec3::new(v.x, v.y, v.z);
		let len = normal.length();
		Self {
			normal: normal / len,
			d: v.w / len,
		}
	}

	/// 判断点是否位于平面负侧 (即视锥外).
	pub fn rejects_point(&self, point: Vec3) -> bool {
		self.normal.dot(point) + self.d < 0.0
	}

	/// 判断 AABB 是否完全位于平面负侧 (即视锥外).
	///
	/// # 判定
	///
	/// 取 AABB 沿平面法线方向最远的顶点 (「正顶点」). 正顶点一旦落在
	/// 平面负侧, 整个 AABB 就都在负侧.
	pub fn rejects_aabb(&self, aabb: &Aabb) -> bool {
		let p = Vec3::new(
			if self.normal.x >= 0.0 {
				aabb.max.x
			} else {
				aabb.min.x
			},
			if self.normal.y >= 0.0 {
				aabb.max.y
			} else {
				aabb.min.y
			},
			if self.normal.z >= 0.0 {
				aabb.max.z
			} else {
				aabb.min.z
			},
		);
		self.rejects_point(p)
	}
}

/// 视锥 (`Frustum`).
///
/// # 六个平面
///
/// 索引顺序为 `0 = left`, `1 = right`, `2 = bottom`, `3 = top`,
/// `4 = near`, `5 = far`, 全部指向视锥内侧.
///
/// # 构造
///
/// 通过从视图投影矩阵提取得到. 调用方传入的矩阵通常由
/// [`crate::camera::Camera::view_projection`] 给出.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
	/// 六个平面, 顺序见类型文档.
	pub planes: [Plane; 6],
}

impl Frustum {
	/// 由视图投影矩阵提取六个平面.
	///
	/// # 输入矩阵
	///
	/// 列主序. `glam` 的 [`Mat4`] 即为列主序 —— 直接传入即可.
	///
	/// # 归一化
	///
	/// 提取出的平面已经归一化 —— `normal` 是单位向量.
	///
	/// # NDC 约定
	///
	/// 假定深度范围为 `[0, 1]` (wgpu / Vulkan). `OpenGL` 的 `[-1, 1]` 需要
	/// 不同的 near 平面公式 —— 见模块文档.
	///
	/// # 行提取
	///
	/// Gribb-Hartmann 方法用的是矩阵的**行**, 而不是列. `glam` 按列主序
	/// 存储 —— `cols[i]` 表示第 i **列**, 而第 i **行**为
	/// `(cols[0][i], cols[1][i], cols[2][i], cols[3][i])` 这样的四元组.
	///
	/// **不要**直接把 `cols[i]` 当作平面: 那样 near 平面会落到相机背后,
	/// 近处物体会被误判为可见.
	pub fn from_view_projection(vp: Mat4) -> Self {
		let cols = vp.to_cols_array_2d();

		// 提取第 i 行.
		let row = |i: usize| {
			Vec4::new(cols[0][i], cols[1][i], cols[2][i], cols[3][i])
		};

		let r0 = row(0);
		let r1 = row(1);
		let r2 = row(2);
		let r3 = row(3);

		Self {
			planes: [
				// left: 由 r3 + r0 得出.
				Plane::from_components(r3 + r0),
				// right: 由 r3 - r0 得出.
				Plane::from_components(r3 - r0),
				// bottom: 由 r3 + r1 得出.
				Plane::from_components(r3 + r1),
				// top: 由 r3 - r1 得出.
				Plane::from_components(r3 - r1),
				// near: 取 r2 (wgpu 约定 `z >= 0`).
				//
				// **不要**改成 `r3 + r2`: 那是 `OpenGL` 的 `[-1, 1]`
				// 约定, 在本项目里它会把 near 平面推到相机后面,
				// 导致近处物体被错误剔除.
				Plane::from_components(r2),
				// far: 由 r3 - r2 得出.
				Plane::from_components(r3 - r2),
			],
		}
	}

	/// 判断点是否位于视锥内.
	///
	/// 六个平面全部通过时才返回 `true`.
	pub fn contains_point(&self, point: Vec3) -> bool {
		!self.planes.iter().any(|p| p.rejects_point(point))
	}

	/// 判断 AABB 是否与视锥相交.
	///
	/// 与 [`Aabb::intersects_frustum`] 等价 —— 两种调用形式并存, 调用方
	/// 按可读性选择.
	pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
		aabb.intersects_frustum(self)
	}
}
