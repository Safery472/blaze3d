//! 坐标轴朝向枚举.
//!
//! [`Direction`][] 表示六个坐标轴朝向, 主要用于贴花映射: 先根据
//! 表面法线确定贴到哪个面, 再按该朝向展开纹理坐标.
//!
//! # 与贴花的关系
//!
//! 贴花纹理默认朝 +Z. 要贴到其他朝向, 需要把贴花旋转到该朝向,
//! [`Direction::rotation`][] 给出对应的四元数.
//!
//! # 映射约定
//!
//! - `Up` / `Down`: Y 轴正 / 负方向.
//! - `South` / `North`: Z 轴正 / 负方向.
//! - `East` / `West`: X 轴正 / 负方向.
//!
//! `rotation` 的具体角度值取决于贴花基底的朝向约定. 若贴花显示
//! 方向不正确, 调整 [`Direction::rotation`][] 的对应分支即可 ——
//! 见该方法的文档.

use glam::{Quat, Vec3};
use std::f32::consts::{FRAC_PI_2, PI};

/// 坐标轴的六个朝向.
///
/// 各变体与坐标轴符号的对应关系见变体文档.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
	/// 竖直向下, 即 -Y 方向.
	Down,
	/// 竖直向上, 即 +Y 方向.
	Up,
	/// 水平朝北, 即 -Z 方向.
	North,
	/// 水平朝南, 即 +Z 方向.
	South,
	/// 水平朝西, 即 -X 方向.
	West,
	/// 水平朝东, 即 +X 方向.
	East,
}

impl Direction {
	/// 取与给定法线最接近的坐标轴.
	///
	/// # 行为
	///
	/// 比较三个分量的绝对值, 最大分量所在的轴即为结果. 分量为 0 时
	/// 按 `< 0.0` 判为负向, 也就是 0 归入正向. 该判定是稳定的, 不会
	/// 因浮点误差在正负之间来回跳.
	///
	/// # 参数
	///
	/// - `normal`: 法线向量, 无需归一化 —— 只比较绝对值.
	pub fn nearest(normal: Vec3) -> Self {
		// 先取绝对值再比较, 判定结果不受向量整体缩放的影响.
		let abs = normal.abs();

		if abs.y >= abs.x && abs.y >= abs.z {
			if normal.y > 0.0 { Self::Up } else { Self::Down }
		} else if abs.z >= abs.x {
			if normal.z > 0.0 { Self::South } else { Self::North }
		} else if normal.x > 0.0 {
			Self::East
		} else {
			Self::West
		}
	}

	/// 把「朝 +Z 的贴花」旋转到本朝向的四元数.
	///
	/// # 注意
	///
	/// **角度是约定而非推导**: 六个分支的取值取决于贴花基底坐标系
	/// 的定义. 渲染出来的贴花方向不对时, 调整对应分支即可 —— 但
	/// **只改一个方向**会破坏它与其他方向的相对关系, 通常需要一起
	/// 重新对齐.
	pub fn rotation(self) -> Quat {
		match self {
			Self::Down => Quat::from_rotation_x(PI),
			Self::Up => Quat::IDENTITY,
			Self::North => Quat::from_rotation_x(-FRAC_PI_2),
			Self::South => Quat::from_rotation_x(FRAC_PI_2),
			Self::West => Quat::from_rotation_y(-FRAC_PI_2),
			Self::East => Quat::from_rotation_y(FRAC_PI_2),
		}
	}

	/// 指向本朝向的单位向量.
	pub const fn normal(self) -> Vec3 {
		match self {
			Self::Down => Vec3::new(0.0, -1.0, 0.0),
			Self::Up => Vec3::new(0.0, 1.0, 0.0),
			Self::North => Vec3::new(0.0, 0.0, -1.0),
			Self::South => Vec3::new(0.0, 0.0, 1.0),
			Self::West => Vec3::new(-1.0, 0.0, 0.0),
			Self::East => Vec3::new(1.0, 0.0, 0.0),
		}
	}

	/// 与本朝向相反的朝向.
	///
	/// 满足 `opposite(opposite(d)) == d`.
	pub const fn opposite(self) -> Self {
		match self {
			Self::Down => Self::Up,
			Self::Up => Self::Down,
			Self::North => Self::South,
			Self::South => Self::North,
			Self::West => Self::East,
			Self::East => Self::West,
		}
	}
}
