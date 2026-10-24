// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 点光源与聚光灯的立方体阴影贴图.
//!
//! [`CubeShadowMap`] 是点光源阴影的核心资源: 一张立方体深度纹理, 从光源
//! 位置朝 6 个方向渲染场景深度. 主渲染 pass 用方向向量采样, 即可得到
//! 任意方向上的最近遮挡深度.
//!
//! # 与方向光阴影的差异
//!
//! 方向光的 [`crate::pipeline::ShadowMap`] 用**正交投影**从单一方向渲染
//! 一张 2D 深度图. 点光与聚光朝四面八方照射, 单一方向覆盖不了 —— 因此
//! 需要立方体贴图的 6 个面 (或 6 张深度图).
//!
//! | | 方向光 `ShadowMap` | 点光 `CubeShadowMap` |
//! |---|---|---|
//! | 投影 | 正交 | 透视 (90° FOV) |
//! | 纹理 | 2D 深度 | 立方体深度 |
//! | 面数 | 1 | 6 |
//! | 采样 | UV + 深度 | 方向向量 |
//! | 光源位置 | 无关 (只关方向) | 关键 |
//!
//! # 聚光灯
//!
//! 聚光灯只照一个锥形区域, 本不需要 6 个面. 但复用立方体贴图最省事:
//! 只渲染锥内的那 1-2 个面即可 (其余面既不渲染, 采样也采不到). 本模块
//! 提供 [`CubeShadowMap::light_view_projection`] 逐个面取矩阵, 调用方
//! 可以只渲染自己需要的面.
//!
//! # 数据流
//!
//! ```text
//! 1. 阴影 pass (每个面一次):
//!    a. 用 light_view_projection(face) 覆盖 globals.view_projection.
//!    b. 渲染场景几何到 target().face_view(face) 作为深度附件.
//!
//! 2. 主渲染 pass:
//!    a. shader 里:
//!       - 世界坐标 → 光源空间方向 dir = world_pos - light_pos.
//!       - 距离 = length(dir).
//!       - reference = (far - near) / distance ... 或直接比较线性深度.
//!       - textureSampleCompareLevel(cube_depth, cube_sampler, dir, reference)
//! ```
//!
//! # WGSL 采样示例
//!
//! ```wgsl
//! @group(3) @binding(0) var shadow_cube: texture_depth_cube;
//! @group(3) @binding(1) var shadow_sampler: sampler_comparison;
//!
//! fn compute_shadow(
//!     light_pos: vec3<f32>,
//!     world_pos: vec3<f32>,
//!     far: f32,
//! ) -> f32 {
//!     let dir = world_pos - light_pos;
//!     let dist = length(dir);
//!     // 方向归一化即可 —— cube 采样只看方向.
//!     let reference = dist / far;
//!     return textureSampleCompareLevel(shadow_cube, shadow_sampler, dir, reference);
//! }
//! ```
//!
//! **注意** 这里用的是 `textureSampleCompareLevel` 而非
//! `textureSampleCompare` —— 前者没有隐式求导, 可以在任意 shader 阶段
//! 使用. cube 的参考值必须与写入端的约定一致 —— 见
//! [`CubeShadowMap::reference_depth`].

use glam::{Mat4, Vec3};

use crate::texture::Sampler;

use super::error::{Error, Result};

/// 立方体阴影贴图采用的深度格式.
///
/// 小尺寸下 `Depth32Float` 的精度够用, 而且所有后端都支持把它建成 cube
/// 视图.
const CUBE_SHADOW_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// 立方体贴图的 6 个面.
///
/// 变体顺序遵循 wgpu / `WebGPU` 的 cube 层约定:
/// `+X, -X, +Y, -Y, +Z, -Z`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeFace {
	/// +X 方向的面.
	PositiveX,
	/// -X 方向的面.
	NegativeX,
	/// +Y 方向的面.
	PositiveY,
	/// -Y 方向的面.
	NegativeY,
	/// +Z 方向的面.
	PositiveZ,
	/// -Z 方向的面.
	NegativeZ,
}

impl CubeFace {
	/// 全部 6 个面, 按层索引升序排列.
	pub const ALL: [CubeFace; 6] = [
		CubeFace::PositiveX,
		CubeFace::NegativeX,
		CubeFace::PositiveY,
		CubeFace::NegativeY,
		CubeFace::PositiveZ,
		CubeFace::NegativeZ,
	];

	/// 该面在立方体贴图中的层索引, 取值范围 `0..6`.
	pub const fn layer(self) -> u32 {
		match self {
			CubeFace::PositiveX => 0,
			CubeFace::NegativeX => 1,
			CubeFace::PositiveY => 2,
			CubeFace::NegativeY => 3,
			CubeFace::PositiveZ => 4,
			CubeFace::NegativeZ => 5,
		}
	}

	/// 该面朝向的单位向量, 位于世界空间.
	pub const fn direction(self) -> Vec3 {
		match self {
			CubeFace::PositiveX => Vec3::X,
			CubeFace::NegativeX => Vec3::NEG_X,
			CubeFace::PositiveY => Vec3::Y,
			CubeFace::NegativeY => Vec3::NEG_Y,
			CubeFace::PositiveZ => Vec3::Z,
			CubeFace::NegativeZ => Vec3::NEG_Z,
		}
	}

	/// 该面的「上」方向, 构造 look-at 视图时使用.
	///
	/// ±Y 面的视线方向本身就沿 Y 轴, 拿 Y 当 up 会与视线平行 —— 改用
	/// Z 兜底.
	pub const fn up(self) -> Vec3 {
		match self {
			CubeFace::PositiveY | CubeFace::NegativeY => Vec3::Z,
			_ => Vec3::Y,
		}
	}
}

// ================ 立方体阴影贴图 ================

/// 供点光源 / 聚光灯使用的立方体阴影贴图.
///
/// # 分辨率
///
/// 每个面都是 `size x size`. 常见取值:
///
/// - 512 —— 开销低; 点光阴影的可见距离有限, 通常够用.
/// - 1024 —— 平衡之选, **推荐默认**.
/// - 2048 —— 画质高, 显存开销达到 2D 阴影图的 6 倍.
///
/// # 光源视锥
///
/// 每个面都用 90° FOV 的透视投影, `near` / `far` 由光源位置与影响范围
/// 决定. 超出 `far` 的物体不参与阴影计算.
pub struct CubeShadowMap {
	/// 存储阴影深度的立方体纹理.
	texture: wgpu::Texture,

	/// 立方体视图 —— 供 shader 采样.
	cube_view: wgpu::TextureView,

	/// 每面的视图 —— 作为 render pass 的深度附件.
	/// 按需创建并缓存.
	face_views: Vec<Option<wgpu::TextureView>>,

	/// 光源的世界坐标.
	light_position: Vec3,

	/// 近平面距离.
	near: f32,

	/// 远平面距离.
	far: f32,

	/// 光源影响范围 (数值与 `far` 相同, 单独存一份让语义更清晰).
	range: f32,

	/// 单个面的分辨率 (正方形边长).
	size: u32,

	/// 阴影比较采样器.
	sampler: Sampler,
}

impl CubeShadowMap {
	/// 创建立方体阴影贴图.
	///
	/// # 参数
	///
	/// - `device` —— 用于创建纹理与采样器.
	/// - `size` —— 每面的分辨率 (正方形边长).
	/// - `light_position` —— 光源的世界坐标.
	/// - `range` —— 光照影响范围, 也就是远平面; 近平面由它推导.
	///
	/// # 近平面
	///
	/// 取 `near = range * 0.01`: 太小会让深度精度都压在近处, 太大又会
	/// 裁掉光源附近的物体. 1% 是常见的折中.
	///
	/// # 错误
	///
	/// - `size == 0` 或超出设备上限 —— [`Error::ZeroSize`] /
	///   [`Error::SizeExceedsLimit`].
	/// - `range <= 0` —— [`Error::InvalidShadowRange`].
	pub fn new(
		device: &wgpu::Device,
		size: u32,
		light_position: Vec3,
		range: f32,
	) -> Result<Self> {
		if size == 0 {
			return Err(Error::ZeroSize {
				width: size,
				height: size,
			});
		}
		let limit = device.limits().max_texture_dimension_2d;
		if size > limit {
			return Err(Error::SizeExceedsLimit {
				requested_width: size,
				requested_height: size,
				limit,
			});
		}
		if !(range.is_finite() && range > 0.0) {
			return Err(Error::InvalidShadowRange);
		}

		let near = range * 0.01;

		let texture = device.create_texture(&wgpu::TextureDescriptor {
			label: Some("cube_shadow_map"),
			size: wgpu::Extent3d {
				width: size,
				height: size,
				depth_or_array_layers: 6,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: CUBE_SHADOW_DEPTH_FORMAT,
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT
				| wgpu::TextureUsages::TEXTURE_BINDING,
			view_formats: &[],
		});

		let cube_view = texture.create_view(&wgpu::TextureViewDescriptor {
			label: Some("cube_shadow_view"),
			dimension: Some(wgpu::TextureViewDimension::Cube),
			..Default::default()
		});

		let sampler = Sampler::comparison(
			device,
			Some("cube_shadow_sampler"),
			wgpu::CompareFunction::LessEqual,
		);

		Ok(Self {
			texture,
			cube_view,
			face_views: vec![None; 6],
			light_position,
			near,
			far: range,
			range,
			size,
			sampler,
		})
	}

	// ================ 光源参数 ================

	/// 返回光源的世界坐标.
	pub fn light_position(&self) -> Vec3 {
		self.light_position
	}

	/// 返回光照影响范围, 数值上等于远平面.
	pub fn range(&self) -> f32 {
		self.range
	}

	/// 返回近平面距离.
	pub fn near(&self) -> f32 {
		self.near
	}

	/// 返回远平面距离.
	pub fn far(&self) -> f32 {
		self.far
	}

	/// 更新光源位置与影响范围.
	///
	/// 深度纹理与采样器都不动 —— 不产生分配.
	pub fn update(&mut self, light_position: Vec3, range: f32) {
		self.light_position = light_position;
		if range.is_finite() && range > 0.0 {
			self.range = range;
			self.far = range;
			self.near = range * 0.01;
		}
	}

	// ================ 光源视角 ================

	/// 返回某个面对应的视图投影矩阵.
	///
	/// # 用途
	///
	/// 渲染该面的阴影之前, 把它写入 globals:
	///
	/// ```ignore
	/// for face in CubeFace::ALL {
	///     system.globals_mut().view_projection =
	///         cube_shadow.light_view_projection(face).to_cols_array_2d();
	///     // 渲染到 cube_shadow.face_view(device, face) 作为深度附件.
	/// }
	/// ```
	pub fn light_view_projection(&self, face: CubeFace) -> Mat4 {
		// 透视投影取 90° FOV —— 立方体的每个面正好覆盖 90°.
		let proj = perspective_rh(
			std::f32::consts::FRAC_PI_2,
			1.0,
			self.near,
			self.far,
		);

		let eye = self.light_position;
		let target = eye + face.direction();
		let view = look_at_rh(eye, target, face.up());

		proj * view
	}

	/// 一次取回全部 6 个面的视图投影矩阵.
	pub fn all_light_view_projections(&self) -> [Mat4; 6] {
		[
			self.light_view_projection(CubeFace::PositiveX),
			self.light_view_projection(CubeFace::NegativeX),
			self.light_view_projection(CubeFace::PositiveY),
			self.light_view_projection(CubeFace::NegativeY),
			self.light_view_projection(CubeFace::PositiveZ),
			self.light_view_projection(CubeFace::NegativeZ),
		]
	}

	/// 返回某面作为 render pass 深度附件的视图.
	///
	/// # 惰性
	///
	/// 某个面首次被请求时才创建视图并缓存; 尺寸不变时后续调用直接返回
	/// 缓存结果.
	pub fn face_view(&mut self, face: CubeFace) -> &wgpu::TextureView {
		let idx = face.layer() as usize;
		if self.face_views[idx].is_none() {
			let view = self.texture.create_view(&wgpu::TextureViewDescriptor {
				label: Some("cube_shadow_face_view"),
				dimension: Some(wgpu::TextureViewDimension::D2),
				base_array_layer: face.layer(),
				array_layer_count: Some(1),
				..Default::default()
			});
			self.face_views[idx] = Some(view);
		}
		self.face_views[idx].as_ref().expect("just inserted")
	}

	/// 返回立方体视图 —— 主 pass 的 shader 用它采样.
	pub fn cube_view(&self) -> &wgpu::TextureView {
		&self.cube_view
	}

	/// 返回比较采样器.
	///
	/// # 类型
	///
	/// 这是一个**比较采样器** —— WGSL 里对应 `sampler_comparison`,
	/// 采样要走 `textureSampleCompareLevel`.
	pub fn sampler(&self) -> &Sampler {
		&self.sampler
	}

	/// 返回底层的立方体深度纹理.
	pub fn raw(&self) -> &wgpu::Texture {
		&self.texture
	}

	/// 返回单个面的分辨率 (正方形边长).
	pub fn size(&self) -> u32 {
		self.size
	}

	/// 参考深度的归一化因子 —— 即 `far`.
	///
	/// # 用途
	///
	/// 写入端 (阴影 pass) 存下的是透视投影之后的非线性深度 `[0, 1]`.
	/// 采样端需要把「世界距离」换算到同一个空间. 最简单的线性近似是
	/// `dist / far`, 但它**只在近平面很小或深度分布均匀时才准确**.
	///
	/// 需要精确匹配时, 调用方应当改用 `light_view_projection(face)`
	/// 自行算出与写入端一致的深度 —— 本函数给出的只是常用近似.
	pub fn reference_depth(&self, world_pos: Vec3) -> f32 {
		let dist = (world_pos - self.light_position).length();
		(dist / self.far).clamp(0.0, 1.0)
	}
}

// ================ 数学辅助函数 ================

/// 构造右手系透视投影矩阵.
///
/// NDC 深度范围是 `[0, 1]` —— 与 wgpu / `Vulkan` 一致, **不是**
/// `OpenGL` 的 `[-1, 1]`.
fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
	let f = 1.0 / (fov_y * 0.5).tan();
	let nf = 1.0 / (near - far);

	Mat4::from_cols(
		glam::Vec4::new(f / aspect, 0.0, 0.0, 0.0),
		glam::Vec4::new(0.0, f, 0.0, 0.0),
		glam::Vec4::new(0.0, 0.0, far * nf, -1.0),
		glam::Vec4::new(0.0, 0.0, far * near * nf, 0.0),
	)
}

/// 构造右手系 look-at 视图矩阵.
///
/// 实现与 [`crate::pipeline::shadow`] 里的同名函数一致 —— 那个是私有
/// 函数, 这里复制一份, 以保持模块之间互不依赖.
fn look_at_rh(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
	let away = (eye - target).normalize();
	let right = up.cross(away).normalize();
	let up = away.cross(right);

	let world = Mat4::from_cols(
		right.extend(0.0),
		up.extend(0.0),
		away.extend(0.0),
		eye.extend(1.0),
	);

	world.inverse()
}
