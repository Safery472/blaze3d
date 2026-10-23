//! Mipmap 链的生成.
//!
//! 本模块提供两条生成路径:
//!
//! - **Compute 路径** (首选): 在同一个 compute pass 内逐级降采样, 每级一次
//!   dispatch —— 从 Mip N-1 读取, 结果写入 Mip N.
//! - **Render pass 路径** (回退): 每级各开一个 render pass, 由全屏四边形
//!   采样上一级.
//!
//! [`Texture::generate_mipmaps`] 会依据纹理格式与用途在两者之间自动挑选.
//!
//! # 维度
//!
//! `D2` / `D3` / `Cube` 都能生成, 但所用路径与 shaders 各不相同:
//!
//! - **D2**: 用 `texture_storage_2d`, dispatch 尺寸为 8x8x1.
//! - **D3**: 用 `texture_storage_3d`, dispatch 尺寸为 4x4x4. 逐级降采样需要
//!   知道第三维的尺寸 —— 该值由 shaders 内的 `textureDimensions` 读出.
//! - **Cube**: 把每个面当作 2D 数组的一层单独处理. 输入视图是
//!   `texture_2d_array`, 输出视图是 `texture_storage_2d_array`, dispatch 的
//!   z 维即面索引 (6 个面 → 6 层).
//!
//! # Cube 的处理方式
//!
//! 立方体贴图的 mipmap **不走** `textureSampleLevel` 按方向采样 —— 那样会在
//! 面边界引入方向插值, 使相邻面共用的棱上出现缝隙. 本模块把 cube 视为 6 层
//! 2D 数组, 各面独立降采样: 棱两侧的像素值分别取自自己的 2x2 邻域, 与 CPU
//! 端盒式滤波的结果一致.
//!
//! # 为什么需要两条路径
//!
//! Compute 路径要求目标 Mip 级能作为 **storage texture** 使用, 即同时满足:
//!
//! - 纹理带 `STORAGE_BINDING` 用途.
//! - 格式本身支持 storage. **sRGB 格式不支持** —— `Rgba8UnormSrgb` /
//!   `Bgra8UnormSrgb` 在 WGSL 里没有对应的 storage 格式.
//!
//! 两条中缺任意一条, 就回退到 render pass 路径.
//!
//! # 采样方式
//!
//! 两条路径都使用双线性过滤. 当采样点与 2x2 纹素精确对齐时, 双线性等价于
//! 对 4 个抽头取平均 (盒式滤波).
//!
//! # 格式绑定
//!
//! Compute 路径的 WGSL storage texture 格式是类型的一部分, 无法参数化: 每种
//! 格式都要有独立的着色器模块与独立的 bind group layout, 视图维度同样算
//! layout 的一部分. 因此 `MipmapGenerator` 在 compute 端的缓存键取
//! (维度, 格式) 二元组, **而非** 单维度或单格式.
//!
//! [`Texture::generate_mipmaps`]: crate::texture::Texture::generate_mipmaps

use std::collections::HashMap;

use crate::system::GpuMesh;
use crate::vertex::{formats, BufferBuilder, DrawMode, VertexConsumer};

use super::error::{Error, Result};
use super::image::{Texture, TextureKind};

// ================ 着色器源码 ================

/// 2D 维度使用的 compute 着色器模板.
///
/// 模板中的 `{STORAGE_FORMAT}` 占位符由
/// [`MipmapGenerator::ensure_compute_pipeline`] 替换成具体的格式名.
const COMPUTE_SHADER_TEMPLATE_2D: &str = include_str!("../../shaders/mipmap.wgsl");

/// 3D 维度使用的 compute 着色器模板.
const COMPUTE_SHADER_TEMPLATE_3D: &str = include_str!("../../shaders/mipmap_3d.wgsl");

/// Cube 维度使用的 compute 着色器模板 (按 2D 数组处理).
const COMPUTE_SHADER_TEMPLATE_CUBE: &str =
	include_str!("../../shaders/mipmap_cube.wgsl");

/// Render pass 路径使用的 WGSL 源码.
///
/// 与 compute 版本的差别在输出形式: 这里由片元着色器返回 `@location(0)` 的
/// 值, 而不是写入 `texture_storage_2d`. 因此本源码不要求 storage 支持 ——
/// 它正是 sRGB 等格式的回退路径.
const RENDER_SHADER_SOURCE: &str = r#"
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(input_texture, input_sampler, in.uv);
}
"#;

// ================ `MipmapGenerator` ================

/// 逐级降采样 Mipmap 的生成器.
///
/// # 生命周期
///
/// 内部持有一个采样器与两套 bind group layout. compute 管线按 (维度, 格式)
/// 惰性创建并缓存; render pass 路径所需的全屏四边形与管线同样是首次用到时
/// 才创建.
///
/// # 缓存键
///
/// compute 端的 layout 与管线都以 (维度, 格式) 为键. **格式必须参与键**:
/// WGSL 的 storage texture 格式是类型的一部分, 若只按维度缓存,
/// `Rgba16Float` 的纹理会拿到声明了 `Rgba8Unorm` 的 layout, 校验会在
/// pipeline 创建时失败.
///
/// # 复用
///
/// 要为多张纹理生成 Mipmap 时, 应当构造一次 generator, 然后反复调用
/// [`Self::generate`]. 只生成一次的场景, 用
/// [`Texture::generate_mipmaps`] 即可.
///
/// [`Texture::generate_mipmaps`]: crate::texture::Texture::generate_mipmaps
pub struct MipmapGenerator {
	device: wgpu::Device,
	queue: wgpu::Queue,

	/// 两条路径共用的输入采样器.
	sampler: wgpu::Sampler,

	// ---- Compute 路径 ----
	//
	// 每个 (维度, 格式) 组合各有一套 layout 与 pipeline: 视图维度是 layout 的
	// 一部分, storage texture 格式也是, 两项都得计入.

	/// 各 (维度, 格式) 组合的 compute bind group layout.
	compute_layouts: HashMap<(TextureKind, wgpu::TextureFormat), wgpu::BindGroupLayout>,

	/// 各 (维度, 格式) 组合的 compute 管线 layout.
	compute_pipeline_layouts:
		HashMap<(TextureKind, wgpu::TextureFormat), wgpu::PipelineLayout>,

	/// 各 (维度, 格式) 组合的 compute 管线.
	compute_pipelines: HashMap<(TextureKind, wgpu::TextureFormat), wgpu::ComputePipeline>,

	// ---- Render 路径 (回退) ----
	//
	// 回退路径只针对 2D 定义 —— 3D / Cube 走 render pass 依赖 attachment 的
	// 视图维度支持, 而且把 render pass 渲染进体纹理在 wgpu 中并非所有后端都
	// 支持. 因此 3D / Cube 遇到不支持 storage 的格式时只能报错.

	/// Render 路径使用的 bind group layout (仅 2D).
	render_bind_group_layout: wgpu::BindGroupLayout,

	/// Render 路径管线的 layout (仅 2D).
	render_pipeline_layout: wgpu::PipelineLayout,

	/// 以格式为键缓存的 render 管线 (仅 2D).
	render_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,

	/// 全屏四边形网格, 首次用到时才创建.
	quad: Option<GpuMesh>,
}

impl MipmapGenerator {
	/// 创建生成器实例.
	///
	/// # 参数
	///
	/// - `device`: 创建管线与缓冲时使用.
	/// - `queue`: 上传全屏四边形顶点数据时使用.
	///
	/// # 惰性
	///
	/// 构造过程只创建采样器与 render 路径的 layout; compute 端的 layout 与
	/// 管线, 以及全屏四边形, 全部按需创建. 首次 [`Self::generate`] 会牵涉管线
	/// 编译, 此后相同 (维度, 格式) 的调用直接复用缓存.
	pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
		let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
			label: Some("mipmap_sampler"),
			address_mode_u: wgpu::AddressMode::ClampToEdge,
			address_mode_v: wgpu::AddressMode::ClampToEdge,
			address_mode_w: wgpu::AddressMode::ClampToEdge,
			mag_filter: wgpu::FilterMode::Linear,
			min_filter: wgpu::FilterMode::Linear,
			mipmap_filter: wgpu::MipmapFilterMode::Nearest,
			..Default::default()
		});

		let render_bind_group_layout =
			device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
				label: Some("mipmap_render_bgl"),
				entries: &[
					wgpu::BindGroupLayoutEntry {
						binding: 0,
						visibility: wgpu::ShaderStages::FRAGMENT,
						ty: wgpu::BindingType::Texture {
							sample_type: wgpu::TextureSampleType::Float {
								filterable: true,
							},
							view_dimension: wgpu::TextureViewDimension::D2,
							multisampled: false,
						},
						count: None,
					},
					wgpu::BindGroupLayoutEntry {
						binding: 1,
						visibility: wgpu::ShaderStages::FRAGMENT,
						ty: wgpu::BindingType::Sampler(
							wgpu::SamplerBindingType::Filtering,
						),
						count: None,
					},
				],
			});

		let render_pipeline_layout =
			device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
				label: Some("mipmap_render_layout"),
				bind_group_layouts: &[Some(&render_bind_group_layout)],
				immediate_size: 0,
			});

		Self {
			device: device.clone(),
			queue: queue.clone(),
			sampler,
			compute_layouts: HashMap::new(),
			compute_pipeline_layouts: HashMap::new(),
			compute_pipelines: HashMap::new(),
			render_bind_group_layout,
			render_pipeline_layout,
			render_pipelines: HashMap::new(),
			quad: None,
		}
	}

	/// 把 Mipmap 生成命令记录进 encoder.
	///
	/// # 路径选择
	///
	/// - 格式支持 storage 且纹理带 `STORAGE_BINDING` 用途: 走 compute 路径.
	/// - 否则纹理带 `RENDER_ATTACHMENT` 用途: 走 render pass 路径.
	/// - 两者都不满足: 返回 [`Error::MissingMipmapUsage`].
	///
	/// # 前置条件
	///
	/// - 需满足 `texture.mip_level_count() >= mip_count`.
	/// - `texture` 带 `TEXTURE_BINDING` 用途.
	/// - 纹理不是 MSAA (`sample_count == 1`).
	/// - mip 0 层已经写入内容.
	///
	/// # 参数
	///
	/// - `encoder`: 命令记录的目标.
	/// - `texture`: 待生成 Mipmap 的纹理.
	/// - `kind`: 纹理维度, `D2` / `D3` / `Cube` 均可.
	/// - `format`: 纹理格式. 既决定走哪条路径, 也是 compute 端 layout 与管线
	///   缓存键的组成部分.
	/// - `base_depth`: mip 0 的深度, 只有 `D3` 会用到 —— 其他维度传 1.
	///   **不能**从 `texture` 读取 —— `wgpu::Texture` 不暴露尺寸, 必须由持有
	///   [`crate::texture::Texture`] 的调用方传入.
	/// - `mip_count`: 生成的级数, 逐级处理 1 到 `mip_count - 1`.
	pub fn generate(
		&mut self,
		encoder: &mut wgpu::CommandEncoder,
		texture: &wgpu::Texture,
		kind: TextureKind,
		format: wgpu::TextureFormat,
		base_depth: u32,
		mip_count: u32,
	) -> Result<()> {
		let usage = texture.usage();

		let compute_ok = storage_format_name(format).is_some()
			&& usage.contains(wgpu::TextureUsages::STORAGE_BINDING);
		let render_ok = usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT);

		if compute_ok {
			self.generate_compute(encoder, texture, kind, format, base_depth, mip_count);
			Ok(())
		} else if kind == TextureKind::D2 && render_ok {
			// render 回退路径只对 2D 有定义 —— 详见 struct 文档.
			self.generate_render(encoder, texture, format, mip_count);
			Ok(())
		} else if render_ok {
			// 3D / Cube 虽带 `RENDER_ATTACHMENT`, 格式却不支持 storage:
			// 没有可用路径, 因此明确报错, 不做静默跳过.
			Err(Error::IncompatibleKind {
				kind,
				reason: "mipmap generation for D3/Cube requires a storage-capable format",
			})
		} else {
			Err(Error::MissingMipmapUsage)
		}
	}

	// ================ Compute 路径 ================

	fn generate_compute(
		&mut self,
		encoder: &mut wgpu::CommandEncoder,
		texture: &wgpu::Texture,
		kind: TextureKind,
		format: wgpu::TextureFormat,
		base_depth: u32,
		mip_count: u32,
	) {
		self.ensure_compute_pipeline(kind, format);

		let key = (kind, format);
		let pipeline = self
			.compute_pipelines
			.get(&key)
			.expect("just inserted");
		let layout = self.compute_layouts.get(&key).expect("just inserted");
		let sampler = &self.sampler;
		let device = &self.device;

		// 降采样要求源视图与目标视图分别是 sampled 视图与 storage 视图, 且视图
		// 维度随 kind 变化 —— 三种 kind 的视图创建参数各不相同.
		let (src_dim, dst_dim) = compute_view_dimensions(kind);

		// 层数: D2 取 1, Cube 取 6.
		let layers = match kind {
			TextureKind::D2 => 1,
			TextureKind::Cube => 6,
			// D3 的层不是数组层 —— 深度范围由 dispatch 的 z 维覆盖.
			TextureKind::D3 => 1,
		};

		// 3D 的 z 维逐级变化, 必须每级单独计算; 2D / Cube 的 z 维就是层数.
		// `base_depth` 只能由调用方传入 —— `wgpu::Texture` 不暴露尺寸.

		let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
			label: Some("mipmap_compute_pass"),
			timestamp_writes: None,
		});

		pass.set_pipeline(pipeline);

		for level in 1..mip_count {
			let (w, h) = mip_dimensions(texture.width(), texture.height(), level);
			let d = mip_depth(base_depth, kind, level);

			// 源视图指向上一级, 目标视图指向当前级, 创建方式如下:
			//
			// - D2 / D3: 单层视图, 维度与纹理自身一致.
			// - Cube: 覆盖全部 6 层的数组视图 —— 一次 dispatch 处理 6 个面.
			let src_view =
				create_mip_view(texture, src_dim, kind, level - 1);
			let dst_view = create_mip_view(texture, dst_dim, kind, level);

			let bind_group = if kind == TextureKind::Cube {
				// Cube 路径不使用采样器, bind group 只有 binding 0 与 2.
				device.create_bind_group(&wgpu::BindGroupDescriptor {
					label: Some("mipmap_compute_bg"),
					layout,
					entries: &[
						wgpu::BindGroupEntry {
							binding: 0,
							resource: wgpu::BindingResource::TextureView(&src_view),
						},
						wgpu::BindGroupEntry {
							binding: 2,
							resource: wgpu::BindingResource::TextureView(&dst_view),
						},
					],
				})
			} else {
				device.create_bind_group(&wgpu::BindGroupDescriptor {
					label: Some("mipmap_compute_bg"),
					layout,
					entries: &[
						wgpu::BindGroupEntry {
							binding: 0,
							resource: wgpu::BindingResource::TextureView(&src_view),
						},
						wgpu::BindGroupEntry {
							binding: 1,
							resource: wgpu::BindingResource::Sampler(sampler),
						},
						wgpu::BindGroupEntry {
							binding: 2,
							resource: wgpu::BindingResource::TextureView(&dst_view),
						},
					],
				})
			};

			pass.set_bind_group(0, &bind_group, &[]);

			// Dispatch 尺寸随 kind 变化:
			//   D2: 8x8x1, z 维取层数 (恒为 1).
			//   D3: 4x4x4, z 维按目标深度计算.
			//   Cube: 8x8, z 维取 6 个面.
			let (wg_x, wg_y, wg_z) = match kind {
				TextureKind::D2 => (w.div_ceil(8), h.div_ceil(8), layers),
				TextureKind::D3 => (w.div_ceil(4), h.div_ceil(4), d.div_ceil(4)),
				TextureKind::Cube => (w.div_ceil(8), h.div_ceil(8), layers),
			};
			pass.dispatch_workgroups(wg_x, wg_y, wg_z);
		}
	}

	/// 在需要时为 `(kind, format)` 建立 compute 管线.
	///
	/// # 缓存键
	///
	/// 键**必须同时包含维度与格式**: 维度决定视图维度, 格式决定 storage texture
	/// 格式, 两者都是 WGSL 类型的一部分. 只按维度缓存, 会让 `Rgba16Float` 的
	/// 纹理拿到声明了 `Rgba8Unorm` 的 layout; 只按格式缓存, 则会让 D2 的
	/// layout 套到 D3 视图上. 两种情况都会在 pipeline 创建时校验失败.
	fn ensure_compute_pipeline(&mut self, kind: TextureKind, format: wgpu::TextureFormat) {
		let key = (kind, format);
		if self.compute_pipelines.contains_key(&key) {
			return;
		}

		let storage_format = storage_format_name(format)
			.expect("compute path requires a storage-capable format");

		// 按维度挑选对应的着色器模板.
		let template = match kind {
			TextureKind::D2 => COMPUTE_SHADER_TEMPLATE_2D,
			TextureKind::D3 => COMPUTE_SHADER_TEMPLATE_3D,
			TextureKind::Cube => COMPUTE_SHADER_TEMPLATE_CUBE,
		};
		let source = template.replace("{STORAGE_FORMAT}", storage_format);

		let shader = self
			.device
			.create_shader_module(wgpu::ShaderModuleDescriptor {
				label: Some("mipmap_compute_shader"),
				source: wgpu::ShaderSource::Wgsl(source.into()),
			});

		// 该 (维度, 格式) 首次出现时, 一并建出它需要的 layout 与 pipeline
		// layout —— 两者都与格式绑定, 都要按双键缓存.
		if !self.compute_layouts.contains_key(&key) {
			let bgl = create_compute_layout(&self.device, kind, format);
			self.compute_layouts.insert(key, bgl);
		}
		if !self.compute_pipeline_layouts.contains_key(&key) {
			let bgl = &self.compute_layouts[&key];
			let pl = self
				.device
				.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
					label: Some("mipmap_compute_layout"),
					bind_group_layouts: &[Some(bgl)],
					immediate_size: 0,
				});
			self.compute_pipeline_layouts.insert(key, pl);
		}

		let pipeline_layout = &self.compute_pipeline_layouts[&key];

		let pipeline =
			self.device
				.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
					label: Some("mipmap_compute_pipeline"),
					layout: Some(pipeline_layout),
					module: &shader,
					entry_point: Some("cs_main"),
					compilation_options:
					wgpu::PipelineCompilationOptions::default(),
					cache: None,
				});

		self.compute_pipelines.insert(key, pipeline);
	}

	// ================ Render 路径 ================

	fn generate_render(
		&mut self,
		encoder: &mut wgpu::CommandEncoder,
		texture: &wgpu::Texture,
		format: wgpu::TextureFormat,
		mip_count: u32,
	) {
		self.ensure_quad();
		self.ensure_render_pipeline(format);

		let quad = self.quad.as_ref().expect("just inserted");
		let pipeline = self
			.render_pipelines
			.get(&format)
			.expect("just inserted");
		let layout = &self.render_bind_group_layout;
		let sampler = &self.sampler;
		let device = &self.device;

		for level in 1..mip_count {
			let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
				label: Some("mipmap_src_view"),
				base_mip_level: level - 1,
				mip_level_count: Some(1),
				base_array_layer: 0,
				array_layer_count: Some(1),
				..Default::default()
			});

			let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
				label: Some("mipmap_dst_view"),
				base_mip_level: level,
				mip_level_count: Some(1),
				base_array_layer: 0,
				array_layer_count: Some(1),
				..Default::default()
			});

			let bind_group =
				device.create_bind_group(&wgpu::BindGroupDescriptor {
					label: Some("mipmap_render_bg"),
					layout,
					entries: &[
						wgpu::BindGroupEntry {
							binding: 0,
							resource: wgpu::BindingResource::TextureView(&src_view),
						},
						wgpu::BindGroupEntry {
							binding: 1,
							resource: wgpu::BindingResource::Sampler(sampler),
						},
					],
				});

			let mut pass =
				encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
					label: Some("mipmap_render_pass"),
					color_attachments: &[Some(wgpu::RenderPassColorAttachment {
						view: &dst_view,
						resolve_target: None,
						depth_slice: None,
						ops: wgpu::Operations {
							load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
							store: wgpu::StoreOp::Store,
						},
					})],
					depth_stencil_attachment: None,
					timestamp_writes: None,
					occlusion_query_set: None,
					multiview_mask: None,
				});

			pass.set_pipeline(pipeline);
			pass.set_bind_group(0, &bind_group, &[]);
			pass.set_vertex_buffer(0, quad.vertex_buffer().slice(..));
			pass.draw(0..quad.vertex_count(), 0..1);
		}
	}

	fn ensure_quad(&mut self) {
		if self.quad.is_none() {
			self.quad = Some(create_fullscreen_quad(&self.device, &self.queue));
		}
	}

	fn ensure_render_pipeline(&mut self, format: wgpu::TextureFormat) {
		if self.render_pipelines.contains_key(&format) {
			return;
		}

		let shader = self
			.device
			.create_shader_module(wgpu::ShaderModuleDescriptor {
				label: Some("mipmap_render_shader"),
				source: wgpu::ShaderSource::Wgsl(RENDER_SHADER_SOURCE.into()),
			});

		let vertex_format = formats::position_tex();
		let vertex_layout = vertex_format.wgpu_layout();
		let buffers = [Some(vertex_layout)];

		let pipeline =
			self.device
				.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
					label: Some("mipmap_render_pipeline"),
					layout: Some(&self.render_pipeline_layout),
					vertex: wgpu::VertexState {
						module: &shader,
						entry_point: Some("vs_main"),
						buffers: &buffers,
						compilation_options:
						wgpu::PipelineCompilationOptions::default(),
					},
					fragment: Some(wgpu::FragmentState {
						module: &shader,
						entry_point: Some("fs_main"),
						targets: &[Some(wgpu::ColorTargetState {
							format,
							blend: None,
							write_mask: wgpu::ColorWrites::ALL,
						})],
						compilation_options:
						wgpu::PipelineCompilationOptions::default(),
					}),
					primitive: wgpu::PrimitiveState {
						topology: wgpu::PrimitiveTopology::TriangleList,
						strip_index_format: None,
						front_face: wgpu::FrontFace::Ccw,
						cull_mode: None,
						polygon_mode: wgpu::PolygonMode::Fill,
						unclipped_depth: false,
						conservative: false,
					},
					depth_stencil: None,
					multisample: wgpu::MultisampleState {
						count: 1,
						mask: !0,
						alpha_to_coverage_enabled: false,
					},
					multiview_mask: None,
					cache: None,
				});

		self.render_pipelines.insert(format, pipeline);
	}
}

// ================ Texture 便利入口 ================

impl Texture {
	/// 生成完整的 Mipmap 链.
	///
	/// # 行为
	///
	/// 以 mip 0 为源, 逐级降采样, 一直生成到 `mip_level_count - 1`.
	///
	/// 路径按格式与用途在内部挑选:
	///
	/// - 格式支持 storage 且纹理带 `STORAGE_BINDING`: compute 路径.
	/// - 否则纹理带 `RENDER_ATTACHMENT`: render pass 路径.
	///
	/// # 维度
	///
	/// `D2` / `D3` / `Cube` 均可. 三种维度各有专属的 compute shaders, 详见
	/// 模块文档.
	///
	/// # 前置条件
	///
	/// - 需满足 `self.mip_level_count() > 1`.
	/// - `self` 带 `TEXTURE_BINDING` 用途.
	/// - 纹理不是 MSAA (`sample_count == 1`).
	/// - mip 0 层已有内容.
	/// - **至少有一条路径可用**:
	///   - compute: 带 `STORAGE_BINDING` 用途且格式支持 storage.
	///   - render: 带 `RENDER_ATTACHMENT` 用途 (**仅 D2**).
	///
	/// # 错误
	///
	/// - 没有多余的 Mip 层: [`Error::NoMipLevels`].
	/// - 两条路径都走不通: [`Error::MissingMipmapUsage`].
	/// - `D3` / `Cube` 遇上不支持 storage 的格式: [`Error::IncompatibleKind`].
	///
	/// # 性能
	///
	/// 每次调用都会新建一个 [`MipmapGenerator`], 首次还牵涉管线编译. 需要反复
	/// 生成时, 请自行缓存一个 `MipmapGenerator`.
	pub fn generate_mipmaps(
		&self,
		device: &wgpu::Device,
		queue: &wgpu::Queue,
	) -> Result<()> {
		let mip_count = self.mip_level_count();
		if mip_count <= 1 {
			return Err(Error::NoMipLevels);
		}

		if !self
			.raw()
			.usage()
			.contains(wgpu::TextureUsages::TEXTURE_BINDING)
		{
			return Err(Error::MissingMipmapUsage);
		}

		let mut generator = MipmapGenerator::new(device, queue);

		let mut encoder =
			device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
				label: Some("mipmap_encoder"),
			});

		generator.generate(
			&mut encoder,
			self.raw(),
			self.kind(),
			self.format(),
			self.depth(),
			mip_count,
		)?;

		queue.submit(Some(encoder.finish()));

		Ok(())
	}
}

// ================ 辅助 ================

/// 求某个 Mip 级的宽高.
///
/// Mip N 的尺寸等于原始尺寸右移 N 位, 且下限为 1.
///
/// **注意**: 原始尺寸为 0 时右移不会变成负数, 结果仍是 0. wgpu 不接受零
/// 尺寸纹理, 正常调用不会产生这种输入; 这里的 `max(1)` 属于防御性处理,
/// 同时也与 wgpu 的 mip 尺寸约定 (始终 >= 1) 对齐.
fn mip_dimensions(width: u32, height: u32, level: u32) -> (u32, u32) {
	((width >> level).max(1), (height >> level).max(1))
}

/// 求某个 Mip 级的深度, 仅对 `D3` 有意义.
///
/// 算法与 [`mip_dimensions`] 相同. 非 `D3` 的纹理一律返回 1.
fn mip_depth(base_depth: u32, kind: TextureKind, level: u32) -> u32 {
	match kind {
		TextureKind::D3 => (base_depth >> level).max(1),
		_ => 1,
	}
}

/// 给出 compute 路径下源视图与目标视图各自的视图维度.
///
/// - `D2`: 源与目标都是 `D2`.
/// - `D3`: 源与目标都是 `D3`.
/// - `Cube`: 源与目标都是 `D2Array` —— 每个面占独立的一层.
fn compute_view_dimensions(
	kind: TextureKind,
) -> (wgpu::TextureViewDimension, wgpu::TextureViewDimension) {
	use wgpu::TextureViewDimension as D;
	match kind {
		TextureKind::D2 => (D::D2, D::D2),
		TextureKind::D3 => (D::D3, D::D3),
		TextureKind::Cube => (D::D2Array, D::D2Array),
	}
}

/// 创建一个只覆盖单个 Mip 级的视图.
///
/// # 参数
///
/// - `dimension`: 视图维度, 由 [`compute_view_dimensions`] 给出.
/// - `kind`: 纹理维度, 决定 `array_layer_count` 的取值.
/// - `level`: 视图指向的 Mip 级.
///
/// # D3 的 depth
///
/// `D3` 视图无需按 depth 切分 —— `base_array_layer` 对 3D 纹理不起作用,
/// 整段深度都落在同一个视图里; 覆盖深度范围是 dispatch 的 z 维的职责.
fn create_mip_view(
	texture: &wgpu::Texture,
	dimension: wgpu::TextureViewDimension,
	kind: TextureKind,
	level: u32,
) -> wgpu::TextureView {
	let (array_layer_count, base_array_layer) = match kind {
		// Cube 一次要处理全部 6 个面, 因此给出 6 层的 `D2Array` 视图.
		TextureKind::Cube => (Some(6), 0),
		// D2: 只取一层.
		TextureKind::D2 => (Some(1), 0),
		// D3: 3D 纹理不存在「数组层」概念, `array_layer_count` 只能是
		// None, 否则 wgpu 校验失败; `base_array_layer` 对 3D 同样无意义.
		TextureKind::D3 => (None, 0),
	};

	texture.create_view(&wgpu::TextureViewDescriptor {
		label: Some("mipmap_view"),
		dimension: Some(dimension),
		base_mip_level: level,
		mip_level_count: Some(1),
		base_array_layer,
		array_layer_count,
		..Default::default()
	})
}

/// 为 (维度, 格式) 创建 compute 路径所需的 bind group layout.
///
/// # 条目
///
/// - binding 0: 采样用的源纹理.
/// - binding 1: 采样器. **Cube 路径不设采样器** —— 它改用 `textureLoad`;
///   该路径的 layout 只含纹理与 storage 两项.
/// - binding 2: 作为写入目标的 storage 纹理.
///
/// # 格式
///
/// `format` 必须与 shaders 里 `{STORAGE_FORMAT}` 替换出来的名字一致 ——
/// WGSL storage texture 格式是类型的一部分, layout 与 shaders 的声明不一致
/// 时, pipeline 创建就会失败. 因此**不要**把它写死成 `Rgba8Unorm`: 那样
/// 只有 `Rgba8Unorm` 纹理能通过, 换成别的格式便会在 pipeline 校验时
/// panic.
fn create_compute_layout(
	device: &wgpu::Device,
	kind: TextureKind,
	format: wgpu::TextureFormat,
) -> wgpu::BindGroupLayout {
	use wgpu::TextureViewDimension as D;

	match kind {
		TextureKind::Cube => {
			// Cube 路径: binding 0 是 `texture_2d_array`, binding 2 是
			// `texture_storage_2d_array`; 不设采样器 (改用 `textureLoad`).
			device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
				label: Some("mipmap_compute_bgl_cube"),
				entries: &[
					wgpu::BindGroupLayoutEntry {
						binding: 0,
						visibility: wgpu::ShaderStages::COMPUTE,
						ty: wgpu::BindingType::Texture {
							sample_type: wgpu::TextureSampleType::Float {
								filterable: true,
							},
							view_dimension: D::D2Array,
							multisampled: false,
						},
						count: None,
					},
					wgpu::BindGroupLayoutEntry {
						binding: 2,
						visibility: wgpu::ShaderStages::COMPUTE,
						ty: wgpu::BindingType::StorageTexture {
							access: wgpu::StorageTextureAccess::WriteOnly,
							format,
							view_dimension: D::D2Array,
						},
						count: None,
					},
				],
			})
		}

		TextureKind::D3 => {
			device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
				label: Some("mipmap_compute_bgl_3d"),
				entries: &[
					wgpu::BindGroupLayoutEntry {
						binding: 0,
						visibility: wgpu::ShaderStages::COMPUTE,
						ty: wgpu::BindingType::Texture {
							sample_type: wgpu::TextureSampleType::Float {
								filterable: true,
							},
							view_dimension: D::D3,
							multisampled: false,
						},
						count: None,
					},
					wgpu::BindGroupLayoutEntry {
						binding: 1,
						visibility: wgpu::ShaderStages::COMPUTE,
						ty: wgpu::BindingType::Sampler(
							wgpu::SamplerBindingType::Filtering,
						),
						count: None,
					},
					wgpu::BindGroupLayoutEntry {
						binding: 2,
						visibility: wgpu::ShaderStages::COMPUTE,
						ty: wgpu::BindingType::StorageTexture {
							access: wgpu::StorageTextureAccess::WriteOnly,
							format,
							view_dimension: D::D3,
						},
						count: None,
					},
				],
			})
		}

		TextureKind::D2 => {
			device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
				label: Some("mipmap_compute_bgl"),
				entries: &[
					wgpu::BindGroupLayoutEntry {
						binding: 0,
						visibility: wgpu::ShaderStages::COMPUTE,
						ty: wgpu::BindingType::Texture {
							sample_type: wgpu::TextureSampleType::Float {
								filterable: true,
							},
							view_dimension: D::D2,
							multisampled: false,
						},
						count: None,
					},
					wgpu::BindGroupLayoutEntry {
						binding: 1,
						visibility: wgpu::ShaderStages::COMPUTE,
						ty: wgpu::BindingType::Sampler(
							wgpu::SamplerBindingType::Filtering,
						),
						count: None,
					},
					wgpu::BindGroupLayoutEntry {
						binding: 2,
						visibility: wgpu::ShaderStages::COMPUTE,
						ty: wgpu::BindingType::StorageTexture {
							access: wgpu::StorageTextureAccess::WriteOnly,
							format,
							view_dimension: D::D2,
						},
						count: None,
					},
				],
			})
		}
	}
}

/// 查出格式在 WGSL 里对应的 storage texture 格式名.
///
/// 返回 `None` 意味着该格式不能作为 storage texture 使用 —— 此类纹理必须
/// 回退到 render pass 路径.
///
/// # sRGB 格式
///
/// **不受支持**. WGSL 里没有 sRGB 的 storage 格式, 所以
/// `Rgba8UnormSrgb` / `Bgra8UnormSrgb` 都不能作为 storage texture.
///
/// # 深度与模板格式
///
/// 同样不支持 —— 它们本就不是 storage texture 的合法格式.
fn storage_format_name(format: wgpu::TextureFormat) -> Option<&'static str> {
	use wgpu::TextureFormat as F;
	match format {
		F::R8Unorm => Some("r8unorm"),
		F::R8Snorm => Some("r8snorm"),
		F::R8Uint => Some("r8uint"),
		F::R8Sint => Some("r8sint"),

		F::R16Uint => Some("r16uint"),
		F::R16Sint => Some("r16sint"),
		F::R16Float => Some("r16float"),

		F::Rg8Unorm => Some("rg8unorm"),
		F::Rg8Snorm => Some("rg8snorm"),
		F::Rg8Uint => Some("rg8uint"),
		F::Rg8Sint => Some("rg8sint"),

		F::R32Uint => Some("r32uint"),
		F::R32Sint => Some("r32sint"),
		F::R32Float => Some("r32float"),

		F::Rg16Uint => Some("rg16uint"),
		F::Rg16Sint => Some("rg16sint"),
		F::Rg16Float => Some("rg16float"),

		F::Rgba8Unorm => Some("rgba8unorm"),
		F::Rgba8Snorm => Some("rgba8snorm"),
		F::Rgba8Uint => Some("rgba8uint"),
		F::Rgba8Sint => Some("rgba8sint"),

		F::Rg32Uint => Some("rg32uint"),
		F::Rg32Sint => Some("rg32sint"),
		F::Rg32Float => Some("rg32float"),

		F::Rgba16Uint => Some("rgba16uint"),
		F::Rgba16Sint => Some("rgba16sint"),
		F::Rgba16Float => Some("rgba16float"),

		F::Rgba32Uint => Some("rgba32uint"),
		F::Rgba32Sint => Some("rgba32sint"),
		F::Rgba32Float => Some("rgba32float"),

		_ => None,
	}
}

/// 构建一个全屏四边形网格.
///
/// 顶点的 Position 已经是 NDC 坐标, UV 覆盖 `(0, 0)` 到 `(1, 1)`.
///
/// # 为什么用 `TriangleList`
///
/// `DrawMode::Quads` 依赖共享索引缓冲, 而那是 `RenderSystem` 的资源.
/// `MipmapGenerator` 与 `RenderSystem` 相互独立, 取不到共享索引, 因此只能用
/// `TriangleList`.
fn create_fullscreen_quad(device: &wgpu::Device, queue: &wgpu::Queue) -> GpuMesh {
	let mut builder = BufferBuilder::with_capacity(6 * 20);
	let mut b = builder.begin(DrawMode::Triangles, formats::position_tex());

	for (x, y, u, v) in [
		(-1.0_f32, -1.0_f32, 0.0_f32, 1.0_f32),
		(1.0, -1.0, 1.0, 1.0),
		(1.0, 1.0, 1.0, 0.0),
		(-1.0, -1.0, 0.0, 1.0),
		(1.0, 1.0, 1.0, 0.0),
		(-1.0, 1.0, 0.0, 0.0),
	] {
		b.vertex(x, y, 0.0).expect("vertex");
		b.uv(u, v).expect("uv");
		b.end_vertex().expect("end_vertex");
	}

	let rendered = b.end();
	GpuMesh::new(device, queue, &rendered)
		.expect("failed to create fullscreen quad")
}
