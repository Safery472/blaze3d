// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 透视相机与透视投影.
//!
//! 投影呈现近大远小, 适用于 3D 场景.
//!
//! # 参数
//!
//! - `fov_y` —— 垂直视场角 (弧度). 常见取值 45° ~ 90°.
//! - `aspect` —— 宽高比, 即 `width / height`.
//! - `near` / `far` —— 近 / 远裁剪面距离. 二者均为正数, 且 `near < far`.
//!
//! # 深度精度
//!
//! 透视投影的深度精度会随距离非线性下降 —— 近处精度高, 远处精度低.
//! `far / near` 比值越大, 远处越容易出现深度冲突 (z-fighting).
//!
//! - 常规场景 —— `near = 0.1`, `far = 1000.0`, 比值为 10000.
//! - 大视距场景 —— `near = 0.5`, `far = 10000.0`, 比值为 20000.
//!
//! **不要**把 `near` 设得太小 (例如 `0.001`) —— 否则远处深度精度
//! 会急剧下降.
//!
//! # 投影矩阵的构造
//!
//! 矩阵为手工构造, 而非调用 `glam` 的投影函数 —— `glam 0.34` 移除了
//! `perspective_rh` 等方法. 由于公式固定, 手工构造反而更清晰:
//!
//! ```text
//! 令 f = 1 / tan(fov_y / 2), nf = near - far
//!
//! [ f/aspect,  0,  0,          0 ]
//! [ 0,         f,  0,          0 ]
//! [ 0,         0,  far/nf,     -1 ]
//! [ 0,         0,  near*far/nf, 0 ]
//! ```
//!
//! 该矩阵把相机空间 z ∈ `[-near, -far]` 映射到 NDC z ∈ `[0, 1]`, 与
//! wgpu / Vulkan 的深度范围一致. **不要**换成 `OpenGL` 约定的 `[-1, 1]`
//! 版本 —— wgpu 会因深度测试错误而渲染异常.

use glam::{Mat3, Mat4, Quat, Vec3, Vec4};

use super::Camera;

/// 透视投影相机.
///
/// # 字段
///
/// 字段全部公开 —— 相机是纯数据, 没有需要维护的不变量. 修改之后, 下一次
/// [`Camera::view_projection`] 调用会反映新的状态.
///
/// # 默认值
///
/// - 位置 —— 原点.
/// - 旋转 —— 单位四元数 (看向 `-Z`).
/// - `fov_y` —— 60° (`FRAC_PI_3`).
/// - `aspect` —— 16:9.
/// - `near` / `far` —— 0.1 / 1000.0.
#[derive(Debug, Clone, Copy)]
pub struct PerspectiveCamera {
	/// 相机在世界空间中的坐标.
	pub position: Vec3,
	/// 相机朝向. `Quat::IDENTITY` 即默认的 `-Z` 朝向.
	pub rotation: Quat,
	/// 垂直视场角, 单位为弧度.
	pub fov_y: f32,
	/// 宽高比, 即 `width / height`.
	pub aspect: f32,
	/// 到近裁剪面的距离.
	pub near: f32,
	/// 到远裁剪面的距离.
	pub far: f32,
}

impl Default for PerspectiveCamera {
	fn default() -> Self {
		Self {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			fov_y: std::f32::consts::FRAC_PI_3, // 即 60°.
			aspect: 16.0 / 9.0,
			near: 0.1,
			far: 1000.0,
		}
	}
}

impl PerspectiveCamera {
	/// 从投影参数创建相机.
	///
	/// 位置取原点, 旋转取单位四元数.
	pub fn new(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
		Self {
			fov_y,
			aspect,
			near,
			far,
			..Default::default()
		}
	}

	// ================ 链式配置 ================

	/// 以链式调用设置相机位置.
	pub fn with_position(mut self, position: Vec3) -> Self {
		self.position = position;
		self
	}

	/// 以链式调用设置相机朝向.
	pub fn with_rotation(mut self, rotation: Quat) -> Self {
		self.rotation = rotation;
		self
	}

	/// 按「看向目标点」设置旋转, 参考点取当前的 `position`.
	///
	/// 这是 [`Self::look_at`] 的链式形式, 旋转规则与退化情况参见该方法.
	pub fn with_look_at(mut self, target: Vec3, up: Vec3) -> Self {
		self.look_at(target, up);
		self
	}

	/// 以链式调用设置垂直视场角 (弧度).
	pub fn with_fov_y(mut self, fov_y: f32) -> Self {
		self.fov_y = fov_y;
		self
	}

	// ================ 运行时更新 ================

	/// 按视口尺寸刷新宽高比.
	///
	/// # 行为
	///
	/// `aspect = width / height`. **宽度或高度为 0** 时保持原值不变 ——
	/// 避免除零产生 `inf` 或 `NaN`.
	///
	/// # 为什么必须拦住 width == 0
	///
	/// `aspect = 0` 会使 [`Camera::projection`] 中的 `f / aspect` 变成
	/// `inf`, 投影矩阵带 `inf` 后着色器输出全为 `NaN`, 画面直接黑掉且
	/// 没有任何报错. 与其在矩阵计算里补防御, 不如在入口处拒绝 —— 零
	/// 宽度视口本身就不是合法输入.
	///
	/// # 时机
	///
	/// 在窗口尺寸变化后调用. 透视投影对宽高比敏感 —— 不更新则画面会被
	/// 拉伸变形.
	pub fn set_viewport(&mut self, width: u32, height: u32) {
		// **不要**只检查 `height`: `width == 0` 时 `aspect = 0`, 投影
		// 矩阵会因此产生 `inf`.
		if width == 0 || height == 0 {
			return;
		}
		self.aspect = width as f32 / height as f32;
	}

	/// 把相机朝向对准目标点.
	///
	/// # 行为
	///
	/// 依据 `position` 与 `target` 的相对方向计算旋转. `up` 负责定义
	/// 「上」方向 —— 通常是 `Vec3::Y`, 但在翻转场景 (例如倒挂) 中可能不同.
	///
	/// # 退化情况
	///
	/// 当 `position == target`, 或 `up` 与视线平行时, 旋转矩阵奇异. 本
	/// 方法不做检查 —— 调用方需保证参数合理. 此时结果未定义, 但不会
	/// panic.
	pub fn look_at(&mut self, target: Vec3, up: Vec3) {
		// 相机朝向 `-Z`, 因此旋转矩阵的 Z 列取「背离目标」的方向.
		//
		// `position` 指向 `target` 的向量表示「朝向目标」, 取反后得到
		// 「背离目标」.
		let away = (self.position - target).normalize();

		// 右方向由 `up` 与 `away` 叉乘得到.
		//
		// 若 `up` 与 `away` 平行, 叉积为零向量, `normalize` 会得到
		// `NaN`. 这属于调用方错误, 不做特殊处理.
		let right = up.cross(away).normalize();

		// 重新计算 up, 保证它与 away, right 正交.
		let up = away.cross(right);

		// 由三个正交基构造旋转.
		//
		// `Mat3::from_cols` 按列填充: 第一列 `right`, 第二列 `up`,
		// 第三列 `away`. 单位正交矩阵即为旋转矩阵.
		self.rotation = Quat::from_mat3(&Mat3::from_cols(right, up, away));
	}
}

impl Camera for PerspectiveCamera {
	fn view(&self) -> Mat4 {
		// 相机的世界变换由旋转与平移组成.
		//
		// `view` 即世界变换的逆矩阵.
		Mat4::from_rotation_translation(self.rotation, self.position).inverse()
	}

	fn projection(&self) -> Mat4 {
		// 手工构造右手系透视投影, `z` 映射到 `[0, 1]` (wgpu 约定).
		//
		// 矩阵形式, 参数含义与验证均见模块文档 —— 此处不再重复.
		let f = 1.0 / (self.fov_y * 0.5).tan();
		let nf = self.near - self.far;

		Mat4::from_cols(
			Vec4::new(f / self.aspect, 0.0, 0.0, 0.0),
			Vec4::new(0.0, f, 0.0, 0.0),
			Vec4::new(0.0, 0.0, self.far / nf, -1.0),
			Vec4::new(0.0, 0.0, self.near * self.far / nf, 0.0),
		)
	}

	fn position(&self) -> Vec3 {
		self.position
	}

	fn forward(&self) -> Vec3 {
		// 相机朝向 `-Z`.
		self.rotation * Vec3::NEG_Z
	}

	fn right(&self) -> Vec3 {
		self.rotation * Vec3::X
	}

	fn up(&self) -> Vec3 {
		self.rotation * Vec3::Y
	}
}
