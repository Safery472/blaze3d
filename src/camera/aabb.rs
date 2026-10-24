// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 轴对齐包围盒 (`Aabb`).
//!
//! [`Aabb`] 是最常用的剔除几何体: 结构简单, 测试快速, 与视锥测试
//! 搭配也最合适. 大多数网格的包围盒在加载时计算一次, 之后保持不变.
//!
//! # 坐标系
//!
//! 沿用 [`crate::camera`] 的约定: 右手系, Y 轴向上.
//!
//! # 用法
//!
//! ```ignore
//! let aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
//! if aabb.intersects_frustum(&frustum) {
//!     // 绘制
//! }
//! ```

use glam::Vec3;

use super::frustum::Frustum;

/// 轴对齐包围盒 (`Aabb`).
///
/// # 字段
///
/// 字段全部公开 —— AABB 是纯数据, 没有需要维护的不变量. `min` 与 `max`
/// 的先后顺序由调用方保证 (`min <= max` 在每一分量上成立).
///
/// # 构造
///
/// - [`Self::new`] —— 直接给定 min / max.
/// - [`Self::from_center_half_extents`] —— 由中心与半长构造.
/// - [`Self::empty`] —— 退化为零体积的盒子.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
	/// 最小顶点 (逐分量取最小).
	pub min: Vec3,
	/// 最大顶点 (逐分量取最大).
	pub max: Vec3,
}

impl Aabb {
	/// 从 `min` / `max` 构造盒子.
	///
	/// **不校验** `min <= max` —— 这一点由调用方保证. 顺序颠倒会让
	/// [`Self::intersects_frustum`] 得出错误判定.
	pub const fn new(min: Vec3, max: Vec3) -> Self {
		Self { min, max }
	}

	/// 由中心与半长构造盒子.
	pub fn from_center_half_extents(center: Vec3, half_extents: Vec3) -> Self {
		Self {
			min: center - half_extents,
			max: center + half_extents,
		}
	}

	/// 构造零体积的盒子 (`min == max == point`).
	pub const fn empty(point: Vec3) -> Self {
		Self {
			min: point,
			max: point,
		}
	}

	/// 返回几何中心.
	pub fn center(&self) -> Vec3 {
		(self.min + self.max) * 0.5
	}

	/// 返回各轴上的半长.
	pub fn half_extents(&self) -> Vec3 {
		(self.max - self.min) * 0.5
	}

	/// 判断包围盒是否与视锥相交.
	///
	/// # 判定
	///
	/// 对每个平面依次测试: 取 AABB 的「正顶点」, 即沿该平面法线方向最远
	/// 的顶点. 若正顶点落在平面的负侧, 说明整个 AABB 都在该平面负侧 ——
	/// 完全位于视锥之外.
	///
	/// 六个平面全部通过时, 判定为相交或包含.
	///
	/// # 保守性
	///
	/// 返回 `true` 只表示「**可能** 与视锥相交」—— 对角落恰好贴住视锥的
	/// 情况同样判为相交. 这是 AABB 剔除固有的保守性: 宁可多画, 不可少画.
	pub fn intersects_frustum(&self, frustum: &Frustum) -> bool {
		for plane in &frustum.planes {
			if plane.rejects_aabb(self) {
				return false;
			}
		}
		true
	}

	/// 判断包围盒是否包含给定点.
	pub fn contains(&self, point: Vec3) -> bool {
		point.cmpge(self.min).all() && point.cmple(self.max).all()
	}

	/// 扩展包围盒, 使其包含给定点.
	pub fn expand_to_include(&mut self, point: Vec3) {
		self.min = self.min.min(point);
		self.max = self.max.max(point);
	}

	/// 返回与另一个 AABB 的并集.
	pub fn union(&self, other: &Aabb) -> Aabb {
		Aabb {
			min: self.min.min(other.min),
			max: self.max.max(other.max),
		}
	}
}

impl Default for Aabb {
	fn default() -> Self {
		Self::empty(Vec3::ZERO)
	}
}
