// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 方向光阴影贴图.
//!
//! [`ShadowMap`] 是一张从光源视角渲染深度的离屏目标. 主渲染 pass 用
//! 比较采样器对它采样, 逐像素还原出阴影因子.
//!
//! # 用途
//!
//! 只服务于方向光的阴影. 光源相机的视角由正交投影给出, 覆盖范围取
//! 场景的包围球 —— 与相机自身的透视投影无关.
//!
//! # 数据流
//!
//! ```text
//! 1. 阴影 pass:
//!    a. 把 light_view_projection 写入 globals.view_projection.
//!    b. render_system.render_frame(shadow_map.target(), |pass| {
//!           渲染场景几何 (只需要顶点位置).
//!       });
//!
//! 2. 主渲染 pass:
//!    a. 把相机 view_projection 写回 globals.
//!    b. shader 里:
//!       - 世界坐标 → 光源 clip space → UV + 深度.
//!       - textureSampleCompare(shadow_tex, shadow_sampler, uv, depth)
//!         得到 [0, 1] 阴影因子.
//! ```
//!
//! # 颜色附件
//!
//! wgpu 的 render pass 允许没有颜色附件, 但 [`Target`] trait 的
//! `color_attachment` 签名固定要求返回一个 attachment —— 本模块因此用
//! `Rgba8Unorm` 充当 dummy 颜色附件, 其内容写入后立即丢弃. 真正的
//! 阴影数据在深度附件里.
//!
//! # 光源方向
//!
//! 这里的方向是**从光源指向场景**, 而不是从场景指向光源. 例如
//! 「太阳在东方, 光向西方照」时, `light_direction` 就是朝西的单位
//! 向量.
//!
//! # WGSL 采样示例
//!
//! ```wgsl
//! @group(3) @binding(0) var shadow_tex: texture_depth_2d;
//! @group(3) @binding(1) var shadow_sampler: sampler_comparison;
//!
//! fn compute_shadow(light_vp: mat4x4<f32>, world_pos: vec3<f32>) -> f32 {
//!     let light_clip = light_vp * vec4<f32>(world_pos, 1.0);
//!     let light_ndc = light_clip.xyz / light_clip.w;
//!     // NDC xy ∈ [-1, 1] → UV ∈ [0, 1].
//!     let uv = vec2<f32>(light_ndc.x * 0.5 + 0.5, 0.5 - light_ndc.y * 0.5);
//!     // **不要**用 light_ndc.z: wgpu NDC z ∈ [0, 1], 已是深度.
//!     return textureSampleCompare(shadow_tex, shadow_sampler, uv, light_ndc.z);
//! }
//! ```
//!
//! [`Target`]: crate::pipeline::Target

use glam::{Mat4, Vec3, Vec4};

use crate::camera::Aabb;
use crate::state::{ColorFormat, DepthFormat};
use crate::texture::Sampler;

use super::error::Result;
use super::target::RenderTarget;

/// 阴影贴图使用的深度格式.
///
/// 选 `Depth32Float`: 精度最高, 并且与其他用途的兼容性最好.
const SHADOW_DEPTH_FORMAT: DepthFormat = DepthFormat::Depth32Float;

/// 阴影贴图占位颜色附件的格式.
///
/// 阴影 pass 并不产生颜色输出 —— 声明这个格式只为满足
/// [`crate::pipeline::Target`] 的签名要求. `Rgba8Unorm` 属于常见格式,
/// 硬件支持面广, 其内容会被丢弃.
const SHADOW_COLOR_FORMAT: ColorFormat = ColorFormat::Rgba8Unorm;

// ================ 阴影贴图 ================

/// 供方向光使用的阴影贴图.
///
/// # 分辨率
///
/// 常见取值:
///
/// - 1024 —— 开销低, 近景画质可以接受.
/// - 2048 —— 平衡之选, **推荐默认**.
/// - 4096 —— 画质高, 但显存与带宽开销都大.
///
/// # 场景包围盒
///
/// 包围盒决定正交投影的覆盖范围. 取小了会切掉视野边缘的阴影; 取大了
/// 又浪费分辨率. 实践上以「相机可见区域 + 场景物体」的联合包围盒效果
/// 最好.
pub struct ShadowMap {
	/// 底层渲染目标. 颜色附件只是 dummy, 阴影数据在深度附件里.
	target: RenderTarget,

	/// 光源相机的视图投影矩阵 (世界空间 → 光源裁剪空间).
	light_view_projection: Mat4,

	/// 光源方向 (世界空间, 由光源指向场景).
	light_direction: Vec3,

	/// 场景包围盒 —— 重算光源视角时用到.
	scene_bounds: Aabb,

	/// 分辨率 (正方形边长).
	size: u32,

	/// 阴影比较采样器.
	///
	/// 用 [`Sampler`] 包装而非裸的 `wgpu::Sampler`, 是为了与
	/// [`crate::texture::BindGroupBuilder::sampler`] 的入参类型一致.
	sampler: Sampler,
}

impl ShadowMap {
	/// 按给定分辨率与场景包围盒创建阴影贴图.
	///
	/// # 参数
	///
	/// - `device` —— 用于创建纹理与采样器.
	/// - `size` —— 分辨率 (正方形), 取值建议见类型文档的「分辨率」.
	/// - `scene_bounds` —— 场景的世界空间包围盒, 决定正交投影范围.
	/// - `light_direction` —— 光源方向 (世界空间, 由光源指向场景).
	///   内部会归一化, 因此传非单位向量也是安全的.
	///
	/// # 错误
	///
	/// - 尺寸为 0 或超出设备上限 —— 由 [`RenderTarget::new`] 返回.
	pub fn new(
		device: &wgpu::Device,
		size: u32,
		scene_bounds: Aabb,
		light_direction: Vec3,
	) -> Result<Self> {
		let target = RenderTarget::new(
			device,
			size,
			size,
			SHADOW_COLOR_FORMAT,
			Some(SHADOW_DEPTH_FORMAT),
			Some("shadow_map"),
		)?;

		let light_view_projection =
			compute_light_view_projection(scene_bounds, light_direction);

		// 比较采样器: 采样时由硬件自动与参考值比较.
		//
		// `CompareFunction::LessEqual` 的语义 —— 纹理深度 <= 参考深度
		// 时返回 1.0 (未被遮挡), 否则返回 0.0 (被遮挡). 这是方向光
		// 阴影的通用约定.
		let sampler = Sampler::comparison(
			device,
			Some("shadow_sampler"),
			wgpu::CompareFunction::LessEqual,
		);

		Ok(Self {
			target,
			light_view_projection,
			light_direction,
			scene_bounds,
			size,
			sampler,
		})
	}

	// ================ 光源视角矩阵 ================

	/// 返回光源视角的视图投影矩阵.
	///
	/// # 用途
	///
	/// 渲染阴影 pass 之前, 把它写入 globals:
	///
	/// ```ignore
	/// let saved = system.globals().view_projection;
	/// system.globals_mut().view_projection =
	///     shadow_map.light_view_projection().to_cols_array_2d();
	///
	/// system.render_frame(shadow_map.target(), |pass| {
	///     pass.set_render_type(&shadow_rt, &shadow_shader, &shadow_layout);
	///     pass.draw_mesh(&scene_mesh);
	/// });
	///
	/// system.globals_mut().view_projection = saved;
	/// ```
	///
	/// # 主渲染 pass
	///
	/// shader 侧同样需要这个矩阵 —— 它把世界坐标投影到光源空间, 供
	/// 深度比较使用. 常见做法是把它作为一个字段并入传给 shader 的
	/// `Globals`.
	pub fn light_view_projection(&self) -> Mat4 {
		self.light_view_projection
	}

	/// 返回光源方向 (世界空间, 由光源指向场景).
	pub fn light_direction(&self) -> Vec3 {
		self.light_direction
	}

	// ================ 采样资源 ================

	/// 返回深度纹理的视图.
	///
	/// # 用途
	///
	/// 与 [`Self::sampler`] 一起装进 bind group, 交给主渲染 pass 的
	/// shader 使用.
	///
	/// # WGSL 声明
	///
	/// ```wgsl
	/// @group(3) @binding(0) var shadow_tex: texture_depth_2d;
	/// ```
	pub fn depth_view(&self) -> &wgpu::TextureView {
		self.target
			.depth_view
			.as_ref()
			.expect("shadow map always has depth attachment")
	}

	/// 返回阴影比较采样器.
	///
	/// # 类型
	///
	/// 这是一个**比较采样器** —— WGSL 里声明为
	/// `sampler_comparison`, layout 里对应
	/// `SamplerBindingType::Comparison`. 采样必须走
	/// `textureSampleCompare`, **不要**用 `textureSample`.
	///
	/// # 绑定
	///
	/// 可以直接交给 [`crate::texture::BindGroupBuilder::sampler`]:
	///
	/// ```ignore
	/// let shadow_bg = BindGroupBuilder::new()
	///     .texture(0, shadow_map.depth_view())
	///     .sampler(1, shadow_map.sampler())
	///     .build(&device, &shadow_layout, Some("shadow_bg"));
	/// ```
	pub fn sampler(&self) -> &Sampler {
		&self.sampler
	}

	// ================ 渲染目标 ================

	/// 返回底层渲染目标.
	///
	/// 渲染阴影 pass 时把它传给 `RenderSystem::render_frame`:
	///
	/// ```ignore
	/// system.render_frame(shadow_map.target(), |pass| { ... });
	/// ```
	pub fn target(&self) -> &RenderTarget {
		&self.target
	}

	// ================ 尺寸与更新 ================

	/// 返回分辨率 (正方形边长).
	pub fn size(&self) -> u32 {
		self.size
	}

	/// 更新场景包围盒与光源方向.
	///
	/// # 行为
	///
	/// 只重算光源视角矩阵; 深度纹理与采样器保持不动 —— 不产生分配.
	///
	/// # 用途
	///
	/// 适合在场景内容变化 (动态加载, 或相机移动后需要扩展覆盖范围)
	/// 时调用. 每帧调用也不会带来重新分配.
	pub fn update(&mut self, scene_bounds: Aabb, light_direction: Vec3) {
		self.scene_bounds = scene_bounds;
		self.light_direction = light_direction;
		self.light_view_projection =
			compute_light_view_projection(scene_bounds, light_direction);
	}

	/// 重设分辨率.
	///
	/// 分辨率变化时会重建深度纹理 —— 有一次分配开销; 分辨率不变时
	/// 什么也不做.
	///
	/// # 错误
	///
	/// 由 [`RenderTarget::resize`] 返回.
	pub fn resize(&mut self, device: &wgpu::Device, size: u32) -> Result<()> {
		if self.size == size {
			return Ok(());
		}

		self.target.resize(device, size, size)?;
		self.size = size;
		Ok(())
	}
}

// ================ 光源视角计算 ================

/// 计算方向光的光源视角矩阵.
///
/// # 算法
///
/// 1. 取包围盒的**包围球** —— 中心 `center`, 半径 `radius` 为对角线
///    长度的一半.
/// 2. 光源相机放在 `center - light_direction * radius * 2`.
/// 3. 正交投影覆盖 `[-radius, radius]` 的正方形.
/// 4. `near = 0.1`, `far = radius * 4` —— 覆盖从相机到场景远端.
///
/// # 为什么用包围球而非包围盒
///
/// 包围球对光源方向不敏感 —— 任意方向都能完整覆盖. 若改用包围盒,
/// 就得按光照方向逐轴投影, 计算更复杂, 而且方向每帧变化都要重算.
///
/// 代价是正交投影比理论上需要的稍大 —— 阴影分辨率会略有浪费.
///
/// # 退化输入
///
/// - 空包围盒 (`min == max`) —— `radius` 用 0.01 兜底, 避免除零.
/// - 零光源方向 —— 用 `-Y` (头顶光) 兜底.
/// - 光源方向接近 ±Y —— `up` 改用 `Z` 而非 `Y`, 避免 `look_at` 退化.
fn compute_light_view_projection(scene_bounds: Aabb, light_direction: Vec3) -> Mat4 {
	let center = scene_bounds.center();
	let half_extents = scene_bounds.half_extents();

	// 包围球半径: 对角线长度的一半. 下限取 0.01, 防止空包围盒退化.
	let radius = half_extents.length().max(0.01);

	// 把光源方向归一化为单位向量.
	let dir = light_direction.try_normalize().unwrap_or(Vec3::NEG_Y);

	// 相机位置: 从场景中心沿光源反向退开.
	let eye = center - dir * radius * 2.0;

	// `look_at` 的 up 轴: 光源方向贴近 Y 轴时改用 Z 轴, 避开退化.
	let up = if dir.dot(Vec3::Y).abs() > 0.99 {
		Vec3::Z
	} else {
		Vec3::Y
	};
	let view = look_at_rh(eye, center, up);

	// 正交投影: 横纵半宽相等, 即正方形视锥.
	let half_size = radius;
	let near = 0.1;
	let far = radius * 4.0;
	let nf = near - far;

	let proj = Mat4::from_cols(
		Vec4::new(1.0 / half_size, 0.0, 0.0, 0.0),
		Vec4::new(0.0, 1.0 / half_size, 0.0, 0.0),
		Vec4::new(0.0, 0.0, 1.0 / nf, 0.0),
		Vec4::new(0.0, 0.0, near / nf, 1.0),
	);

	proj * view
}

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
fn look_at_rh(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
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
