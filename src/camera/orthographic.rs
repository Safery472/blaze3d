//! 正交相机与正交投影.
//!
//! 投影保持远近同大, 适用于 2D 场景, 工程图, 或有意压缩深度的 3D 场景.
//!
//! # 与透视相机的区别
//!
//! - **无近大远小**: 物体的投影尺寸只由其世界尺寸决定.
//! - **无 `fov_y`**: 视野范围由 [`OrthographicCamera::height`] 与
//!   [`OrthographicCamera::aspect`] 决定.
//! - **深度线性**: 深度精度在整个范围内均匀分布.
//!
//! # 视野的两种表达
//!
//! - [`OrthographicCamera::height`] + [`OrthographicCamera::aspect`] —— 给出
//!   垂直世界尺寸与宽高比. 相机始终居中于 `position`, 视野随 `aspect`
//!   变化.
//! - 直接操作投影矩阵 —— 需要非对称视野 (例如滚动地图) 时, 手工构造矩阵.
//!
//! # 投影矩阵的构造
//!
//! 矩阵为手工构造, 而非调用 `glam` 的投影函数 —— `glam 0.34` 移除了
//! `orthographic_rh` 等方法. 公式固定如下:
//!
//! ```text
//! 令 half_w = height * aspect / 2
//!    half_h = height / 2
//!    nf = near - far (负数)
//!
//! [ 1/half_w,  0,         0,      0 ]
//! [ 0,         1/half_h,  0,      0 ]
//! [ 0,         0,         1/nf,   0 ]
//! [ 0,         0,         near/nf, 1 ]
//! ```
//!
//! 该矩阵负责把相机空间 z ∈ `[-near, -far]` 映射为 NDC z ∈ `[0, 1]`.

use glam::{Mat4, Quat, Vec3, Vec4};

use super::Camera;

/// 正交投影相机.
///
/// # 字段
///
/// 字段全部公开.
///
/// # 默认值
///
/// - 位置 —— 原点.
/// - 旋转 —— 单位四元数 (看向 `-Z`).
/// - `height` —— 2.0.
/// - `aspect` —— 16:9.
/// - `near` / `far` —— 0.1 / 1000.0.
#[derive(Debug, Clone, Copy)]
pub struct OrthographicCamera {
	/// 相机在世界空间中的坐标.
	pub position: Vec3,
	/// 相机朝向. `Quat::IDENTITY` 即默认的 `-Z` 朝向.
	pub rotation: Quat,
	/// 可见区域在竖直方向上的尺寸 (世界单位).
	///
	/// 视野中心位于 `position`, 竖直方向覆盖 `position.y - height/2`
	/// 到 `position.y + height/2` 之间的世界坐标.
	pub height: f32,
	/// 宽高比, 即 `width / height`.
	pub aspect: f32,
	/// 到近裁剪面的距离.
	pub near: f32,
	/// 到远裁剪面的距离.
	pub far: f32,
}

impl Default for OrthographicCamera {
	fn default() -> Self {
		Self {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			height: 2.0,
			aspect: 16.0 / 9.0,
			near: 0.1,
			far: 1000.0,
		}
	}
}

impl OrthographicCamera {
	/// 从视野尺寸与裁剪面参数创建相机.
	///
	/// 位置取原点, 旋转取单位四元数.
	pub fn new(height: f32, aspect: f32, near: f32, far: f32) -> Self {
		Self {
			height,
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

	/// 以链式调用设置垂直视野尺寸.
	pub fn with_height(mut self, height: f32) -> Self {
		self.height = height;
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
	/// `aspect = 0` 会使 [`Camera::projection`] 中的 `half_width` 变成 0,
	/// 进而 `1.0 / half_width` 得到 `inf`. 投影矩阵一旦带 `inf`, 着色器
	/// 输出便全是 `NaN`, 画面直接黑掉且没有任何报错.
	pub fn set_viewport(&mut self, width: u32, height: u32) {
		// **不要**只检查 `height`: `width == 0` 时 `aspect = 0`, 投影
		// 矩阵会因此产生 `inf`.
		if width == 0 || height == 0 {
			return;
		}
		self.aspect = width as f32 / height as f32;
	}
}

impl Camera for OrthographicCamera {
	fn view(&self) -> Mat4 {
		Mat4::from_rotation_translation(self.rotation, self.position).inverse()
	}

	fn projection(&self) -> Mat4 {
		// 根据 `height` 与 `aspect` 推导四个裁剪面.
		//
		// 相机空间的原点位于视野中心, 因此左右上下均关于原点对称.
		let half_height = self.height * 0.5;
		let half_width = half_height * self.aspect;
		let nf = self.near - self.far;

		Mat4::from_cols(
			Vec4::new(1.0 / half_width, 0.0, 0.0, 0.0),
			Vec4::new(0.0, 1.0 / half_height, 0.0, 0.0),
			Vec4::new(0.0, 0.0, 1.0 / nf, 0.0),
			Vec4::new(0.0, 0.0, self.near / nf, 1.0),
		)
	}

	fn position(&self) -> Vec3 {
		self.position
	}

	fn forward(&self) -> Vec3 {
		self.rotation * Vec3::NEG_Z
	}

	fn right(&self) -> Vec3 {
		self.rotation * Vec3::X
	}

	fn up(&self) -> Vec3 {
		self.rotation * Vec3::Y
	}
}
