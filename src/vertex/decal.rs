// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 贴花纹理坐标生成器.
//!
//! [`SheetedDecalTextureGenerator`][] 是一个顶点消费者装饰器: 它把
//! 「世界坐标 + 法线」的写入转换成「贴花 UV」的写入, 从而把一张贴花
//! 纹理以任意朝向贴到表面上.
//!
//! # 工作方式
//!
//! 装饰器会**暂存**每次写入的位置与法线, 等到
//! [`VertexConsumer::end_vertex`][] 时再:
//!
//! 1. 先用 `normal_inverse_pose` 变换法线, 把它归入 6 个轴向之一.
//! 2. 再用 `camera_inverse_pose` 变换位置, 并按该轴向旋转到局部坐标.
//! 3. 把局部 xy 乘以 `texture_scale`, 得到 UV.
//! 4. 最后把完整顶点一次性转发给下游.
//!
//! 颜色与 UV 都由装饰器决定 (分别是固定白色与计算值), 调用方为这两项
//! 传入的参数会被忽略.
//!
//! # 与即时转发的区别
//!
//! [`crate::vertex::MultiConsumer`][] 的每个方法都立即转发. 本装饰器
//! 必须先暂存 —— UV 要等法线与位置都到齐之后才能计算. 因此 `vertex` /
//! `normal` 只更新内部状态, 真正的转发发生在 `end_vertex`.

use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
use std::f32::consts::{FRAC_PI_2, PI};

use super::consumer::VertexConsumer;
use super::direction::Direction;
use super::error::Result;

const DEFAULT_OVERLAY_U: i32 = 0;
const DEFAULT_OVERLAY_V: i32 = 10;
/// 默认光照坐标, 是一个打包值: 两个 16 位分量都取最大光照.
const DEFAULT_LIGHT_COORDS: i32 = 15728880;

/// 把世界坐标 + 法线转换成贴花 UV 的装饰器.
///
/// # 生命周期
///
/// 装饰器独占持有下游 [`VertexConsumer`][] 的引用. 在它存活期间, 下游
/// 同样处于独占借用状态 —— 这既是 Rust 借用规则的必然结果, 也是设计
/// 意图: 同一条下游不应被两条写入路径同时持有.
pub struct SheetedDecalTextureGenerator<'a> {
	delegate: &'a mut dyn VertexConsumer,

	/// 相机位姿矩阵的逆.
	///
	/// 只在构造时求一次逆, 之后每次 `end_vertex` 都复用它, 省掉了每写
	/// 一个顶点就求一次逆的开销.
	///
	/// **构造之后不要修改** —— 它只对应构造那一刻的位姿.
	camera_inverse_pose: Mat4,

	/// 法线位姿矩阵的逆.
	///
	/// 与 `camera_inverse_pose` 相同, 只在构造时求一次逆.
	normal_inverse_pose: Mat3,

	/// UV 缩放系数, 取值越大贴花越密集.
	texture_scale: f32,

	// ---- 单次顶点的暂存状态 ----
	//
	// 这些字段会在每次 `end_vertex` 之后由 `reset_state` 重置为默认值.

	x: f32,
	y: f32,
	z: f32,

	overlay_u: i32,
	overlay_v: i32,

	light_coords: i32,

	nx: f32,
	ny: f32,
	nz: f32,
}

impl<'a> SheetedDecalTextureGenerator<'a> {
	/// 构造一个装饰器.
	///
	/// # 参数
	///
	/// - `delegate`: 接收转发结果的下游消费者.
	/// - `camera_pose`: 相机位姿矩阵. 内部会对它求逆, 调用方传入正向
	///   矩阵即可.
	/// - `normal_pose`: 法线位姿矩阵, 同样在内部求逆.
	/// - `texture_scale`: UV 缩放系数, 取值越大贴花越密集.
	pub fn new(
		delegate: &'a mut dyn VertexConsumer,
		camera_pose: Mat4,
		normal_pose: Mat3,
		texture_scale: f32,
	) -> Self {
		Self {
			delegate,
			camera_inverse_pose: camera_pose.inverse(),
			normal_inverse_pose: normal_pose.inverse(),
			texture_scale,
			x: 0.0,
			y: 0.0,
			z: 0.0,
			overlay_u: DEFAULT_OVERLAY_U,
			overlay_v: DEFAULT_OVERLAY_V,
			light_coords: DEFAULT_LIGHT_COORDS,
			nx: 0.0,
			ny: 1.0,
			nz: 0.0,
		}
	}

	/// 把暂存状态恢复为默认值.
	///
	/// 每次 `end_vertex` 结束时都会调用. 若调用方在写下一个顶点时没有
	/// 提供某个字段, 该字段就沿用它在这里被填入的默认值.
	fn reset_state(&mut self) {
		self.x = 0.0;
		self.y = 0.0;
		self.z = 0.0;
		self.overlay_u = DEFAULT_OVERLAY_U;
		self.overlay_v = DEFAULT_OVERLAY_V;
		self.light_coords = DEFAULT_LIGHT_COORDS;
		self.nx = 0.0;
		self.ny = 1.0;
		self.nz = 0.0;
	}
}

impl VertexConsumer for SheetedDecalTextureGenerator<'_> {
	/// 暂存位置, 不立即转发给下游.
	fn vertex(&mut self, x: f32, y: f32, z: f32) -> Result<()> {
		self.x = x;
		self.y = y;
		self.z = z;
		Ok(())
	}

	/// 忽略传入的颜色. 贴花固定使用白色, 交由下游的默认颜色或着色器
	/// 处理.
	///
	/// **这里返回 `Ok` 而不是错误**: 调用方通常会按完整顶点协议依次
	/// 调用 `vertex` / `color` / `uv` 等, 而装饰器只是跳过自己不需要
	/// 的字段, 没有报错的理由.
	fn color(&mut self, _r: u8, _g: u8, _b: u8, _a: u8) -> Result<()> {
		Ok(())
	}

	/// 忽略传入的 UV, 贴花 UV 由装饰器自行计算.
	fn uv(&mut self, _u: f32, _v: f32) -> Result<()> {
		Ok(())
	}

	/// 暂存覆盖层坐标, 不立即转发给下游.
	fn overlay_coords(&mut self, u: i32, v: i32) -> Result<()> {
		self.overlay_u = u;
		self.overlay_v = v;
		Ok(())
	}

	/// 暂存打包后的光照坐标.
	///
	/// 两个入参各是一个 16 位分量, 内部会把它们打包进一个 32 位值,
	/// 语义与 [`VertexConsumer::uv2`][] 一致.
	fn uv2(&mut self, u: i32, v: i32) -> Result<()> {
		self.light_coords = (u & 0xFFFF) | ((v & 0xFFFF) << 16);
		Ok(())
	}

	/// 暂存法线, 不立即转发给下游.
	fn normal(&mut self, x: f32, y: f32, z: f32) -> Result<()> {
		self.nx = x;
		self.ny = y;
		self.nz = z;
		Ok(())
	}

	/// 忽略骨骼索引, 贴花不使用骨骼.
	fn bone_index(&mut self, _indices: [u8; 4]) -> Result<()> {
		Ok(())
	}

	/// 忽略骨骼权重, 原因同样是贴花不使用骨骼.
	fn bone_weight(&mut self, _weights: [u8; 4]) -> Result<()> {
		Ok(())
	}

	/// 计算 UV, 并把一个完整顶点转发给下游.
	///
	/// # 计算流程
	///
	/// 1. 法线先经 `normal_inverse_pose` 变换, 再归入 6 个轴向之一.
	/// 2. 位置经 `camera_inverse_pose` 变换为齐次坐标, 随后按
	///    `-Y` → `-X` → 该轴向的顺序施加旋转.
	/// 3. 把旋转后坐标的 xy 取负, 再乘以 `texture_scale` 得到 UV.
	/// 4. 转发时颜色固定为白色, UV 取计算值, 其余字段使用暂存值.
	fn end_vertex(&mut self) -> Result<()> {
		// 步骤 1: 把法线归入某个轴向.
		//
		// 法线先经过 `normal_inverse_pose` 变换, 然后取最接近的轴.
		// 变换后的法线未必归一化, 但 `Direction::nearest` 只比较
		// 绝对值, 判定结果不受影响.
		let world_normal = self.normal_inverse_pose * Vec3::new(self.nx, self.ny, self.nz);
		let direction = Direction::nearest(world_normal);

		// 步骤 2: 变换位置, 并施加对齐旋转.
		//
		// 位置先被变换到相机局部空间. 齐次坐标的 w 分量不参与之后的
		// 旋转.
		let world_pos = self.camera_inverse_pose * Vec4::new(self.x, self.y, self.z, 1.0);

		// 三次旋转按调用顺序依次左乘: q3 * q2 * q1 * v.
		//
		// 这套对齐顺序是固定约定:
		//   1. 绕 Y 轴转 180°, 贴花朝向由 +Z 变为 -Z.
		//   2. 绕 X 轴转 -90°, 贴花平面对齐到 XY 平面.
		//   3. 按所在轴向旋转.
		//
		// **不要**调整这三个旋转的顺序或角度 —— 它们共同定义了贴花的
		// 坐标系. 贴花方向不对时, 应改 `Direction::rotation`, 而不是
		// 这里.
		let align = direction.rotation()
			* Quat::from_rotation_x(-FRAC_PI_2)
			* Quat::from_rotation_y(PI);

		let local = align * Vec3::new(world_pos.x, world_pos.y, world_pos.z);

		// 步骤 3: 由局部坐标算出 UV.
		//
		// 取负号是为了让贴花方向与表面朝向一致 —— 具体的朝向仍由上面
		// 的对齐约定决定.
		let u = -local.x * self.texture_scale;
		let v = -local.y * self.texture_scale;

		// 步骤 4: 把完整顶点转发给下游.
		//
		// 调用顺序遵循完整顶点协议: `vertex` → `color` → `uv` →
		// `overlay` → `uv2` → `normal` → `end_vertex`. 下游会按自身的
		// 格式校验该顺序.
		self.delegate.vertex(self.x, self.y, self.z)?;
		// 贴花颜色固定为白色; 需要 tint 时, 由调用方在上游完成着色.
		self.delegate.color(255, 255, 255, 255)?;
		self.delegate.uv(u, v)?;
		self.delegate.overlay_coords(self.overlay_u, self.overlay_v)?;
		self.delegate.uv2(
			self.light_coords & 0xFFFF,
			(self.light_coords >> 16) & 0xFFFF,
		)?;
		self.delegate.normal(self.nx, self.ny, self.nz)?;
		self.delegate.end_vertex()?;

		self.reset_state();
		Ok(())
	}

	/// 把默认颜色设置转发给下游.
	///
	/// 装饰器自身虽然忽略颜色, 下游却可能用到它, 因此这里无条件转发.
	fn default_color(&mut self, r: u8, g: u8, b: u8, a: u8) {
		self.delegate.default_color(r, g, b, a);
	}

	/// 把「取消默认颜色」的调用转发给下游.
	fn unset_default_color(&mut self) {
		self.delegate.unset_default_color();
	}
}
