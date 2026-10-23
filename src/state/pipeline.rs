//! 管线状态描述与管线缓存.
//!
//! 模块内容分两类:
//!
//! - **状态描述**: [`RenderType`] 及其构建器 [`RenderTypeBuilder`], 外加
//!   混合, 深度, 剔除等各类状态枚举.
//! - **管线缓存**: [`PipelineDesc`] / [`ComputePipelineDesc`] 充当缓存
//!   key, [`PipelineCache`] 依它们创建并复用 [`wgpu::RenderPipeline`] /
//!   [`wgpu::ComputePipeline`].
//!
//! # 合并到单个文件的原因
//!
//! [`PipelineDesc`] 直接持有 [`RenderType`] —— 二者是「key 与内容」的
//! 关系. 拆成两个文件会让阅读时反复跳转; 合在一起后, 读完一个文件即可
//! 掌握管线状态的表达方式与缓存机制.
//!
//! # 状态枚举的设计
//!
//! 每个枚举对应 wgpu 的一个状态字段, 以 Rust 枚举列举有限选项. 与直接
//! 暴露 wgpu 类型相比, 收益有三点:
//!
//! - **减少误用**: `BlendMode::Off` 对应 `None`, `BlendMode::Alpha`
//!   对应一组固定的 `BlendState` —— 调用方不必记住 `src_factor` /
//!   `dst_factor` / `operation` 的正确组合.
//! - **默认值明确**: [`RenderTypeBuilder::new`] 给出合理的默认状态,
//!   调用方只覆盖自己关心的字段.
//! - **`Hash` / `Eq`**: 所有枚举与 [`RenderType`] 都实现了 `Hash + Eq`,
//!   因而能充当缓存 key. 例外是 `RenderType` 的 `name` 字段 —— 见该
//!   类型的「相等性」一节.
//!
//! # 与 [`VertexFormat`] 的关系
//!
//! [`RenderType`] 内嵌一个 [`VertexFormat`] —— 顶点格式属于管线状态的
//! 组成部分 (决定 `VertexBufferLayout`). 改动格式即意味着换一条管线.
//!
//! # 缓存不设上限
//!
//! [`PipelineCache`] 没有淘汰策略 —— 这里假设 [`RenderType`] 的组合
//! 数量有限. **一旦支持动态生成的顶点格式**, 缓存就会无界增长. 届时
//! 需要重新评估: 加入 LRU, 或为 `VertexFormat` 设计带指纹的 key.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::vertex::{DrawMode, VertexFormat};

// ================ 混合模式枚举 ================

/// 片元颜色与附件已有内容的组合方式.
///
/// 该枚举最终映射到 `ColorTargetState` 的 `blend` 字段. 只想不透明覆盖
/// 时选 [`Self::Off`]; 需要与背景叠加时, 按叠加意图挑选其余变体.
///
/// # 等价变体
///
/// [`Self::Alpha`] 与 [`Self::Translucent`] 的**混合公式完全一致** ——
/// 拆成两个变体只为语义清晰, 代价是缓存副作用 (详见变体文档).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendMode {
	/// 关闭混合 —— 片元直接覆盖附件内容.
	Off,
	/// 标准 alpha 混合, 按源 alpha 在源与目标之间插值.
	///
	/// 公式为 `src * src_alpha + dst * (1 - src_alpha)`.
	Alpha,
	/// 加性混合, 在目标颜色上累加源颜色.
	///
	/// 公式为 `src * src_alpha + dst`. 常用于发光体与粒子.
	Additive,
	/// 乘性混合, 把源与目标颜色相乘.
	///
	/// 公式为 `src * dst`. 可用于阴影叠加与滤色.
	Multiply,
	/// 公式与 [`Self::Alpha`] 相同, 命名上强调「半透明」语义.
	///
	/// 之所以保留为独立变体: 调用方写 `BlendMode::Translucent` 时, 意图
	/// 比写 `Alpha` 更直白.
	///
	/// # 缓存副作用
	///
	/// **两个变体算出不同的 [`RenderType`] 哈希** —— [`PipelineDesc`]
	/// 会把它们当作两条独立管线分别编译, 而底层 `BlendState` 一模一样,
	/// 因此这部分编译纯属冗余.
	///
	/// 若场景中大量使用 `Translucent` 且在意管线缓存命中率, 统一改用
	/// [`Self::Alpha`]; 若更看重代码可读性, 保持现状也无妨 —— 多编译
	/// 一条管线的代价很小.
	///
	/// [`RenderType`]: crate::state::RenderType
	/// [`PipelineDesc`]: crate::state::PipelineDesc
	Translucent,
}

impl BlendMode {
	/// 把本变体转换成 wgpu 的混合状态.
	///
	/// [`Self::Off`] 返回 `None`, 表示管线关闭混合; 其余变体返回对应的
	/// [`wgpu::BlendState`].
	///
	/// # 公式
	///
	/// 各变体的混合公式写在其变体文档里. 此处四个 `BlendFactor` 参数交给
	/// `blend_state` 组装成 `BlendState`, `operation` 一律取 `Add`.
	///
	/// # `Alpha` 与 `Translucent`
	///
	/// 两者共用一个 `match` 分支 —— 公式本来就相同, 见 [`BlendMode`] 的
	/// 「等价变体」一节.
	pub fn wgpu_blend(self) -> Option<wgpu::BlendState> {
		use wgpu::BlendFactor;

		match self {
			Self::Off => None,

			// src * src_alpha + dst * (1 - src_alpha)
			Self::Alpha | Self::Translucent => Some(blend_state(
				BlendFactor::SrcAlpha,
				BlendFactor::OneMinusSrcAlpha,
				BlendFactor::One,
				BlendFactor::OneMinusSrcAlpha,
			)),

			// src * src_alpha + dst
			Self::Additive => Some(blend_state(
				BlendFactor::SrcAlpha,
				BlendFactor::One,
				BlendFactor::One,
				BlendFactor::One,
			)),

			// src * dst
			Self::Multiply => Some(blend_state(
				BlendFactor::Dst,
				BlendFactor::Zero,
				BlendFactor::One,
				BlendFactor::OneMinusSrcAlpha,
			)),
		}
	}
}

/// 组装一个 [`wgpu::BlendState`].
///
/// # 参数
///
/// - **`color_src`** / **`color_dst`**: 颜色分量的源因子与目标因子.
/// - **`alpha_src`** / **`alpha_dst`**: alpha 分量的源因子与目标因子.
///
/// # 固定值
///
/// 颜色与 alpha 两个分量的 `operation` 都固定为
/// [`wgpu::BlendOperation::Add`] —— 所有预设混合模式一律采用加法组合,
/// 差别只在因子.
///
/// 需要减法或极值运算时, 请直接手工构造 [`wgpu::BlendState`].
fn blend_state(
	color_src: wgpu::BlendFactor,
	color_dst: wgpu::BlendFactor,
	alpha_src: wgpu::BlendFactor,
	alpha_dst: wgpu::BlendFactor,
) -> wgpu::BlendState {
	use wgpu::{BlendComponent, BlendOperation};

	wgpu::BlendState {
		color: BlendComponent {
			src_factor: color_src,
			dst_factor: color_dst,
			operation: BlendOperation::Add,
		},
		alpha: BlendComponent {
			src_factor: alpha_src,
			dst_factor: alpha_dst,
			operation: BlendOperation::Add,
		},
	}
}

// ================ 深度比较函数 ================

/// 深度测试所用的比较函数.
///
/// 与 [`wgpu::CompareFunction`] 一一对应, 变体命名也沿用 wgpu 的写法.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
	/// 永不通过 —— 比较结果恒为假.
	Never,
	/// 新值小于已有值时通过.
	Less,
	/// 新值等于已有值时通过.
	Equal,
	/// 新值小于或等于已有值时通过.
	LessEqual,
	/// 新值大于已有值时通过.
	Greater,
	/// 新值不等于已有值时通过.
	NotEqual,
	/// 新值大于或等于已有值时通过.
	GreaterEqual,
	/// 恒通过 —— 比较结果恒为真, 效果等同于关闭深度测试.
	Always,
}

impl CompareOp {
	/// 返回对应的 wgpu 比较函数.
	pub const fn wgpu(self) -> wgpu::CompareFunction {
		match self {
			Self::Never => wgpu::CompareFunction::Never,
			Self::Less => wgpu::CompareFunction::Less,
			Self::Equal => wgpu::CompareFunction::Equal,
			Self::LessEqual => wgpu::CompareFunction::LessEqual,
			Self::Greater => wgpu::CompareFunction::Greater,
			Self::NotEqual => wgpu::CompareFunction::NotEqual,
			Self::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
			Self::Always => wgpu::CompareFunction::Always,
		}
	}
}

// ================ 深度测试与写入 ================

/// 深度测试与深度写入的组合模式.
///
/// 三个字段合起来描述完整的深度行为. 之所以用结构体而不是枚举: 这些
/// 维度相互正交 —— 例如「只写入, 不测试」同样是一种合法状态.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthMode {
	/// 是否启用深度测试.
	pub test: bool,
	/// 是否把深度写回附件.
	pub write: bool,
	/// 测试所用的比较函数.
	pub compare: CompareOp,
}

impl DepthMode {
	/// 关掉全部深度行为.
	///
	/// 适用于 2D, UI, 后处理等不依赖深度的场景.
	pub const fn disabled() -> Self {
		Self {
			test: false,
			write: false,
			compare: CompareOp::Always,
		}
	}

	/// 标准深度配置: 测试 + 写入, 比较函数为 `LessEqual`.
	///
	/// 大多数 3D 渲染都用这一档: 「较近者胜」, 且允许共面片元通过.
	pub const fn read_write() -> Self {
		Self {
			test: true,
			write: true,
			compare: CompareOp::LessEqual,
		}
	}

	/// 只读深度: 执行测试, 但不写入.
	///
	/// 适合透明物体 —— 需要靠深度测试隐藏被遮挡的部分, 又不写深度,
	/// 以免干扰后续绘制.
	pub const fn read_only() -> Self {
		Self {
			test: true,
			write: false,
			compare: CompareOp::LessEqual,
		}
	}
}

// ================ 三角形剔除 ================

/// 三角形的剔除模式.
///
/// 决定是否丢弃背面或正面朝向的三角形, 对应
/// `wgpu::PrimitiveState::cull_mode` 字段.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CullMode {
	/// 不做剔除, 正反两面都绘制.
	None,
	/// 丢弃背面朝向的三角形.
	///
	/// 这是默认值. 闭合几何体的内表面永远看不见, 丢掉它们可以省下
	/// 光栅化开销.
	Back,
	/// 丢弃正面朝向的三角形.
	///
	/// 适合需要看到几何体内部的场景 (例如倒影).
	Front,
}

impl CullMode {
	/// 转换成 wgpu 的 `cull_mode` 取值.
	///
	/// `CullMode::None` 映射为 `None`; 其余变体各自对应一个
	/// [`wgpu::Face`].
	pub fn wgpu_face(self) -> Option<wgpu::Face> {
		match self {
			Self::None => None,
			Self::Back => Some(wgpu::Face::Back),
			Self::Front => Some(wgpu::Face::Front),
		}
	}
}

// ================ 颜色与深度的写入开关 ================

/// 颜色与深度的写入开关.
///
/// 典型用途: 阴影贴图只要深度 (「只写深度」), 后期叠加只要颜色
/// (「只写颜色」).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WriteMask {
	/// 是否允许写颜色附件.
	pub color: bool,
	/// 是否允许写深度附件.
	pub depth: bool,
}

impl WriteMask {
	/// 颜色与深度都允许写入.
	pub const ALL: Self = Self {
		color: true,
		depth: true,
	};

	/// 只允许写颜色.
	pub const COLOR_ONLY: Self = Self {
		color: true,
		depth: false,
	};

	/// 只允许写深度.
	pub const DEPTH_ONLY: Self = Self {
		color: false,
		depth: true,
	};
}

// ================ 多边形光栅化 ================

/// 三角形的光栅化填充方式.
///
/// # 特性门控
///
/// [`Self::Line`] 依赖 [`wgpu::Features::POLYGON_MODE_LINE`]. 使用前必须
/// 先检查设备特性 —— 设备不支持时, 创建管线会 panic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolygonMode {
	/// 正常填充三角形, 也是默认值.
	Fill,
	/// 只绘制三角形的边线.
	///
	/// **依赖特性**: `POLYGON_MODE_LINE`.
	Line,
}

impl PolygonMode {
	/// 返回对应的 wgpu 多边形模式.
	pub fn wgpu_mode(self) -> wgpu::PolygonMode {
		match self {
			Self::Fill => wgpu::PolygonMode::Fill,
			Self::Line => wgpu::PolygonMode::Line,
		}
	}
}

// ================ 颜色附件格式 ================

/// 颜色附件的纹理格式.
///
/// 这里只收录常用格式. 若要支持其他格式, 新增变体并在 [`Self::wgpu`]
/// 中补齐映射即可.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorFormat {
	/// 每通道 8 位的 BGRA, 非线性 (未做 sRGB 编码).
	///
	/// Windows 平台上很常见. 使用它时 gamma 需要自行处理.
	Bgra8Unorm,
	/// 每通道 8 位的 BGRA, 带 sRGB 编码.
	///
	/// 采样时自动反 gamma, 写入时自动做 gamma 编码. 窗口表面最常用的
	/// 格式.
	Bgra8UnormSrgb,
	/// 每通道 8 位的 RGBA, 非线性.
	Rgba8Unorm,
	/// 每通道 8 位的 RGBA, 带 sRGB 编码.
	Rgba8UnormSrgb,
	/// 每通道 16 位浮点.
	///
	/// 适合 HDR 渲染, 也适合对精度要求更高的中间目标.
	Rgba16Float,
}

impl ColorFormat {
	/// 返回对应的 wgpu 纹理格式.
	pub const fn wgpu(self) -> wgpu::TextureFormat {
		match self {
			Self::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
			Self::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
			Self::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
			Self::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
			Self::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
		}
	}

	/// 从 wgpu 纹理格式反查.
	///
	/// 仅能识别 [`Self`] 已覆盖的格式, 其余一律返回 `None`.
	///
	/// # 用途
	///
	/// [`crate::pipeline::SurfaceFrame::acquire`] 需要把
	/// [`wgpu::SurfaceConfiguration::format`] 转成 [`ColorFormat`].
	/// 表面格式由显示系统决定, 可能落在本枚举之外 —— 此时返回 `None`,
	/// 究竟是扩展枚举还是另选格式, 由调用方决定.
	pub fn from_wgpu(format: wgpu::TextureFormat) -> Option<Self> {
		match format {
			wgpu::TextureFormat::Bgra8Unorm => Some(Self::Bgra8Unorm),
			wgpu::TextureFormat::Bgra8UnormSrgb => Some(Self::Bgra8UnormSrgb),
			wgpu::TextureFormat::Rgba8Unorm => Some(Self::Rgba8Unorm),
			wgpu::TextureFormat::Rgba8UnormSrgb => Some(Self::Rgba8UnormSrgb),
			wgpu::TextureFormat::Rgba16Float => Some(Self::Rgba16Float),
			_ => None,
		}
	}
}

// ================ 深度附件格式 ================

/// 深度附件的纹理格式.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DepthFormat {
	/// 16 位归一化深度值.
	Depth16Unorm,
	/// 24 位深度, 具体存储方式由实现自行决定.
	Depth24Plus,
	/// 24 位深度外加 8 位模板.
	Depth24PlusStencil8,
	/// 32 位浮点深度.
	///
	/// 精度在所有变体中最高, 适合大视距或对深度精度敏感的场景.
	Depth32Float,
}

impl DepthFormat {
	/// 返回对应的 wgpu 纹理格式.
	pub const fn wgpu(self) -> wgpu::TextureFormat {
		match self {
			Self::Depth16Unorm => wgpu::TextureFormat::Depth16Unorm,
			Self::Depth24Plus => wgpu::TextureFormat::Depth24Plus,
			Self::Depth24PlusStencil8 => wgpu::TextureFormat::Depth24PlusStencil8,
			Self::Depth32Float => wgpu::TextureFormat::Depth32Float,
		}
	}
}

// ================ 渲染状态组合 ================

/// 描述一次绘制所需要的全部管线状态.
///
/// # 相等性
///
/// 实现了 `PartialEq + Eq + Hash`, 因此能作为 [`PipelineDesc`] 的字段
/// 参与缓存查找.
///
/// **`name` 不参与相等性比较** —— 处理方式与 [`VertexFormat`] 对待元素
/// 名一致: `name` 只服务于调试与 wgpu 管线的 `label`, 状态完全相同的
/// 两个 `RenderType` 理应命中同一条缓存. **切勿**把 `name` 加回
/// `PartialEq` / `Hash` —— 否则「名字不同而状态相同」的组合会各编译
/// 一条管线, 白白消耗显存与编译时间.
///
/// # 与 `DrawMode` 的关系
///
/// `mode` 字段既决定 `PrimitiveState.topology`, 也决定绘制时走不走索引.
/// [`DrawMode::Quads`] 会被映射为 `TriangleList`, 索引由
/// [`crate::system::SharedIndexBuffers`] 提供.
#[derive(Debug, Clone)]
pub struct RenderType {
	/// 供调试使用的名称.
	///
	/// **不参与** `PartialEq` / `Hash`, 原因见类型文档.
	pub name: &'static str,
	/// 顶点数据的格式.
	pub vertex_format: VertexFormat,
	/// 图元组装方式.
	pub mode: DrawMode,
	/// 颜色混合方式.
	pub blend: BlendMode,
	/// 深度测试与写入方式.
	pub depth: DepthMode,
	/// 背面剔除方式.
	pub cull: CullMode,
	/// 颜色 / 深度写入掩码.
	pub write_mask: WriteMask,
	/// 多边形填充方式.
	pub polygon_mode: PolygonMode,
	/// 颜色附件的格式.
	pub color_format: ColorFormat,
	/// 深度附件的格式; `None` 表示不挂深度附件.
	pub depth_format: Option<DepthFormat>,
	/// MSAA 采样数; 取 1 即关闭 MSAA.
	pub sample_count: u32,
	/// 是否在顶点上传阶段完成排序.
	///
	/// 为 `true` 时, 绘制前需要先给图元排序 (典型例子是半透明四边形),
	/// 并应调用 [`crate::vertex::Building::set_quad_sorting`].
	pub sort_on_upload: bool,
}

impl PartialEq for RenderType {
	fn eq(&self, other: &Self) -> bool {
		self.vertex_format == other.vertex_format
			&& self.mode == other.mode
			&& self.blend == other.blend
			&& self.depth == other.depth
			&& self.cull == other.cull
			&& self.write_mask == other.write_mask
			&& self.polygon_mode == other.polygon_mode
			&& self.color_format == other.color_format
			&& self.depth_format == other.depth_format
			&& self.sample_count == other.sample_count
			&& self.sort_on_upload == other.sort_on_upload
	}
}

impl Eq for RenderType {}

impl Hash for RenderType {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.vertex_format.hash(state);
		self.mode.hash(state);
		self.blend.hash(state);
		self.depth.hash(state);
		self.cull.hash(state);
		self.write_mask.hash(state);
		self.polygon_mode.hash(state);
		self.color_format.hash(state);
		self.depth_format.hash(state);
		self.sample_count.hash(state);
		self.sort_on_upload.hash(state);
	}
}

impl RenderType {
	/// 为指定名称创建 [`RenderTypeBuilder`].
	pub fn builder(name: &'static str) -> RenderTypeBuilder {
		RenderTypeBuilder::new(name)
	}
}

// ================ 渲染状态构建器 ================

/// 逐步构造 [`RenderType`] 的构建器.
///
/// # 默认值
///
/// 各项默认值都按「典型 3D 渲染」来定:
///
/// - 顶点格式: **无** —— 必须由 [`Self::vertex_format`] 显式给出.
/// - 图元模式: 三角形列表.
/// - 混合: 关闭, 即不透明.
/// - 深度: 测试 + 写入, 比较函数 `LessEqual`.
/// - 剔除: 背面.
/// - 写入: 颜色与深度都写.
/// - 多边形: 填充.
/// - 颜色格式: `Bgra8UnormSrgb`.
/// - 深度格式: `Depth32Float`.
/// - 采样数: 1, 即不启用 MSAA.
/// - 排序: 关闭.
pub struct RenderTypeBuilder {
	name: &'static str,
	vertex_format: Option<VertexFormat>,
	mode: DrawMode,
	blend: BlendMode,
	depth: DepthMode,
	cull: CullMode,
	write_mask: WriteMask,
	polygon_mode: PolygonMode,
	color_format: ColorFormat,
	depth_format: Option<DepthFormat>,
	sample_count: u32,
	sort_on_upload: bool,
}

impl RenderTypeBuilder {
	/// 以给定名称创建构建器, 其余字段取默认值.
	pub fn new(name: &'static str) -> Self {
		Self {
			name,
			vertex_format: None,
			mode: DrawMode::Triangles,
			blend: BlendMode::Off,
			depth: DepthMode::read_write(),
			cull: CullMode::Back,
			write_mask: WriteMask::ALL,
			polygon_mode: PolygonMode::Fill,
			color_format: ColorFormat::Bgra8UnormSrgb,
			depth_format: Some(DepthFormat::Depth32Float),
			sample_count: 1,
			sort_on_upload: false,
		}
	}

	/// 指定顶点格式.
	pub fn vertex_format(mut self, f: VertexFormat) -> Self {
		self.vertex_format = Some(f);
		self
	}

	/// 指定图元组装方式.
	pub fn mode(mut self, m: DrawMode) -> Self {
		self.mode = m;
		self
	}

	/// 指定颜色混合方式.
	pub fn blend(mut self, b: BlendMode) -> Self {
		self.blend = b;
		self
	}

	/// 指定深度测试与写入方式.
	pub fn depth(mut self, d: DepthMode) -> Self {
		self.depth = d;
		self
	}

	/// 指定背面剔除方式.
	pub fn cull(mut self, c: CullMode) -> Self {
		self.cull = c;
		self
	}

	/// 指定颜色 / 深度写入掩码.
	pub fn write_mask(mut self, w: WriteMask) -> Self {
		self.write_mask = w;
		self
	}

	/// 指定多边形填充方式.
	pub fn polygon_mode(mut self, p: PolygonMode) -> Self {
		self.polygon_mode = p;
		self
	}

	/// 指定颜色附件格式.
	pub fn color_format(mut self, f: ColorFormat) -> Self {
		self.color_format = f;
		self
	}

	/// 指定深度附件格式; 传 `None` 表示不挂深度附件.
	pub fn depth_format(mut self, f: Option<DepthFormat>) -> Self {
		self.depth_format = f;
		self
	}

	/// 指定 MSAA 采样数.
	pub fn sample_count(mut self, n: u32) -> Self {
		self.sample_count = n;
		self
	}

	/// 指定是否在上传阶段完成排序.
	pub fn sort_on_upload(mut self, s: bool) -> Self {
		self.sort_on_upload = s;
		self
	}

	/// 结束构建, 产出 [`RenderType`].
	///
	/// # Panics
	///
	/// 顶点格式缺失时 panic. 这是刻意设计 —— 顶点格式不存在合理的
	/// 默认值, 忘记设置属于编程错误, 而非运行期才出现的条件.
	pub fn build(self) -> RenderType {
		RenderType {
			name: self.name,
			vertex_format: self
				.vertex_format
				.expect("vertex_format is required"),
			mode: self.mode,
			blend: self.blend,
			depth: self.depth,
			cull: self.cull,
			write_mask: self.write_mask,
			polygon_mode: self.polygon_mode,
			color_format: self.color_format,
			depth_format: self.depth_format,
			sample_count: self.sample_count,
			sort_on_upload: self.sort_on_upload,
		}
	}
}

// ================ 渲染管线 key ================

/// 渲染管线缓存的 key.
///
/// 三个字段合起来唯一确定一条 [`wgpu::RenderPipeline`]:
///
/// - **`render_type`**: 顶点格式, 混合, 深度等状态.
/// - **`shader_id`**: 顶点 / 片元着色器, 由
///   [`crate::shader::Shader::id`] 提供.
/// - **`layout_id`**: 管线布局 (即 `bind group` 排列), 由
///   [`crate::state::LayoutHandle::id`] 提供.
///
/// # 实现 `Hash + Eq`
///
/// 要充当 [`HashMap`] 的 key, 这两个 trait 缺一不可. `RenderType` 的
/// 全部**功能字段**都参与, 唯独 `name` 例外 —— 见 [`RenderType`] 的
/// 「相等性」一节.
///
/// # 等价状态
///
/// 有些 `RenderType` 字段组合会产出**完全一样**的 wgpu 管线, 例如
/// [`BlendMode::Alpha`] 与 [`BlendMode::Translucent`]. 但它们在
/// `PipelineDesc` 里是**两个不同的 key**, 于是各自编译一条管线. 这是
/// 为状态表达清晰而付出的缓存命中率代价.
///
/// [`BlendMode::Alpha`]: crate::state::BlendMode::Alpha
/// [`BlendMode::Translucent`]: crate::state::BlendMode::Translucent
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineDesc {
	/// 完整的管线状态.
	pub render_type: RenderType,
	/// 着色器标识.
	pub shader_id: u64,
	/// 布局标识.
	pub layout_id: u64,
}

// ================ 计算管线 key ================

/// 计算管线缓存的 key.
///
/// 两个字段合起来唯一确定一条 [`wgpu::ComputePipeline`]:
///
/// - **`shader_id`**: 计算着色器, 由 [`crate::shader::Shader::id`] 提供.
/// - **`layout_id`**: 管线布局, 由 [`crate::state::LayoutHandle::id`]
///   提供.
///
/// # 与 `PipelineDesc` 的差异
///
/// 计算管线不涉及 [`RenderType`] —— 顶点格式, 混合, 深度, 剔除这一套
/// 光栅化状态统统用不上, 只剩「用哪份着色器 + 哪个布局」两个维度.
///
/// # 入口点
///
/// 入口点固定为 `"cs_main"`, 与渲染管线的 `"vs_main"` / `"fs_main"`
/// 约定保持一致. 需要自定义入口点时, 得先扩展本类型.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComputePipelineDesc {
	/// 着色器标识.
	pub shader_id: u64,
	/// 布局标识.
	pub layout_id: u64,
}

// ================ 管线缓存表 ================

/// 按 key 复用渲染与计算管线的缓存.
///
/// # 用途
///
/// 创建 [`wgpu::RenderPipeline`] / [`wgpu::ComputePipeline`] 都相当昂贵
/// (要编译着色器, 还要做状态验证), 同一组合反复创建会直接拖慢帧率.
/// 本缓存以 [`PipelineDesc`] / [`ComputePipelineDesc`] 为 key, 复用已经
/// 建好的管线.
///
/// # 两张表
///
/// 渲染与计算使用不同的缓存 key, 故分开存放. 两者的生命周期和查询方式
/// 虽然一致, 但 key 语义不同 —— 硬合并成一张表得额外包一层枚举, 并不
/// 划算.
///
/// # 计数方法
///
/// 两类管线各自配备一组计数方法:
///
/// - [`Self::render_len`] / [`Self::render_is_empty`] —— 渲染管线.
/// - [`Self::compute_len`] / [`Self::compute_is_empty`] —— 计算管线.
///
/// 刻意**不提供**笼统的 `len` —— 否则只添加过计算管线的调用方会读到
/// `0` 而做出错误判断. 确实需要总数时把两者相加即可.
///
/// # 不设上限
///
/// 缓存没有容量上限. 对多数应用而言管线组合数是有限的 (几十到几百
/// 种), 用不着淘汰策略. **一旦支持动态生成的顶点格式**, 缓存就会无界
/// 增长 —— 那时需要引入 LRU, 或限制 [`RenderType`] 的种类数.
///
/// # 与 [`crate::state::LayoutHandle`] 的关系
///
/// 两种 key 的布局部分都取自 `LayoutHandle` 的 `layout_id`; 真正创建管线
/// 时, 再通过 `LayoutHandle::raw()` 取出 [`wgpu::PipelineLayout`].
pub struct PipelineCache {
	device: Arc<wgpu::Device>,
	render_cache: HashMap<PipelineDesc, Arc<wgpu::RenderPipeline>>,
	compute_cache: HashMap<ComputePipelineDesc, Arc<wgpu::ComputePipeline>>,
}

impl PipelineCache {
	/// 创建一个空缓存.
	pub fn new(device: Arc<wgpu::Device>) -> Self {
		Self {
			device,
			render_cache: HashMap::new(),
			compute_cache: HashMap::new(),
		}
	}

	/// 查询渲染管线, 不存在时创建并登记.
	///
	/// # 行为
	///
	/// - 命中: 直接返回已有管线的 `Arc`.
	/// - 未命中: 新建管线, 写入缓存后再返回.
	///
	/// # 参数
	///
	/// - **`desc`**: 缓存 key, 携带完整的管线状态描述.
	/// - **`shader`**: 顶点 / 片元着色器模块.
	/// - **`layout`**: 管线布局.
	///
	/// # Panics
	///
	/// 出现下列情形时, wgpu 会在 `create_render_pipeline` 内部 panic:
	///
	/// - 顶点格式的 `shader_location` 与着色器的 `@location` 对不上.
	/// - 着色器编译出错 —— 该错误要到创建管线时才暴露.
	/// - 用了 `PolygonMode::Line`, 而设备没启用 `POLYGON_MODE_LINE`.
	/// - 状态组合非法 (例如 MSAA 采样数与附件对不上).
	///
	/// 这些都属于编程错误, 应当在开发阶段暴露, 而不是留到运行时降级
	/// 处理.
	pub fn get_or_create(
		&mut self,
		desc: &PipelineDesc,
		shader: &wgpu::ShaderModule,
		layout: &wgpu::PipelineLayout,
	) -> Arc<wgpu::RenderPipeline> {
		if let Some(pipeline) = self.render_cache.get(desc) {
			return Arc::clone(pipeline);
		}

		let rt = &desc.render_type;

		// 顶点缓冲布局.
		//
		// 顶点格式为空时 (例如全屏三角形) 不绑定任何顶点缓冲.
		//
		// `VertexFormat::new` 已经保证格式非空, 所以实际执行时走不到
		// 「空」这一分支 —— 保留该判断只是让逻辑能应对未来可能出现的
		// 空格式.
		//
		// **生命周期处理**: `non_empty_storage` 声明在 if 表达式之外,
		// 生命周期因而覆盖整个函数体; `vertex_buffers` 借用它, 在
		// `create_render_pipeline` 调用期间始终有效.
		let non_empty_storage;
		let vertex_buffers: &[Option<wgpu::VertexBufferLayout>] =
			if rt.vertex_format.elements().is_empty() {
				&[]
			} else {
				non_empty_storage = [Some(rt.vertex_format.wgpu_layout())];
				&non_empty_storage
			};

		let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some(rt.name),
			layout: Some(layout),
			vertex: wgpu::VertexState {
				module: shader,
				entry_point: Some("vs_main"),
				buffers: vertex_buffers,
				compilation_options: wgpu::PipelineCompilationOptions::default(),
			},
			fragment: Some(wgpu::FragmentState {
				module: shader,
				entry_point: Some("fs_main"),
				targets: &[Some(wgpu::ColorTargetState {
					format: rt.color_format.wgpu(),
					blend: rt.blend.wgpu_blend(),
					write_mask: if rt.write_mask.color {
						wgpu::ColorWrites::ALL
					} else {
						wgpu::ColorWrites::empty()
					},
				})],
				compilation_options: wgpu::PipelineCompilationOptions::default(),
			}),
			primitive: wgpu::PrimitiveState {
				topology: rt.mode.wgpu_topology(),
				strip_index_format: None,
				front_face: wgpu::FrontFace::Ccw,
				cull_mode: rt.cull.wgpu_face(),
				polygon_mode: rt.polygon_mode.wgpu_mode(),
				unclipped_depth: false,
				conservative: false,
			},
			depth_stencil: rt.depth_format.map(|format| wgpu::DepthStencilState {
				format: format.wgpu(),
				// 同时满足两个条件才写深度: 写入掩码允许, 且深度模式允许.
				//
				// wgpu 30 起 `depth_write_enabled` 的类型是
				// `Option<bool>` —— `Some(true)` 开启写入, `Some(false)`
				// 关闭写入.
				depth_write_enabled: Some(rt.write_mask.depth && rt.depth.write),
				// 深度测试关闭时不能给比较函数, 这是 wgpu 的一致性要求.
				depth_compare: if rt.depth.test {
					Some(rt.depth.compare.wgpu())
				} else {
					None
				},
				stencil: wgpu::StencilState::default(),
				bias: wgpu::DepthBiasState::default(),
			}),
			multisample: wgpu::MultisampleState {
				count: rt.sample_count,
				mask: !0,
				alpha_to_coverage_enabled: false,
			},
			multiview_mask: None,
			cache: None,
		});

		let pipeline = Arc::new(pipeline);
		self.render_cache.insert(desc.clone(), Arc::clone(&pipeline));
		pipeline
	}

	/// 查询计算管线, 不存在时创建并登记.
	///
	/// # 行为
	///
	/// - 命中: 直接返回已有管线的 `Arc`.
	/// - 未命中: 新建管线, 写入缓存后再返回.
	///
	/// # 参数
	///
	/// - **`desc`**: 缓存 key, 只含 `shader_id` 与 `layout_id`.
	/// - **`shader`**: 计算着色器模块.
	/// - **`layout`**: 管线布局.
	///
	/// # 入口点
	///
	/// 固定为 `"cs_main"`, 与渲染管线的 `"vs_main"` / `"fs_main"` 约定
	/// 保持一致.
	///
	/// # Panics
	///
	/// 出现下列情形时, wgpu 会在 `create_compute_pipeline` 内部 panic:
	///
	/// - 着色器编译出错.
	/// - 入口点名字不是 `cs_main`.
	/// - 布局与着色器声明的 `bind group` 不匹配.
	///
	/// 与渲染版一样, 这些都属于编程错误.
	pub fn get_or_create_compute(
		&mut self,
		desc: &ComputePipelineDesc,
		shader: &wgpu::ShaderModule,
		layout: &wgpu::PipelineLayout,
	) -> Arc<wgpu::ComputePipeline> {
		if let Some(pipeline) = self.compute_cache.get(desc) {
			return Arc::clone(pipeline);
		}

		let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: Some("blaze3d_compute_pipeline"),
			layout: Some(layout),
			module: shader,
			entry_point: Some("cs_main"),
			compilation_options: wgpu::PipelineCompilationOptions::default(),
			cache: None,
		});

		let pipeline = Arc::new(pipeline);
		self.compute_cache.insert(desc.clone(), Arc::clone(&pipeline));
		pipeline
	}

	/// 清空渲染与计算两张缓存表.
	///
	/// 已经交出去的 `Arc` 不受影响 —— 调用方手上的引用依然有效; 只是
	/// 之后的查询会重新创建管线.
	pub fn clear(&mut self) {
		self.render_cache.clear();
		self.compute_cache.clear();
	}

	/// 渲染管线缓存中的条目数.
	///
	/// **只统计**渲染管线; 计算管线那一半请用 [`Self::compute_len`].
	pub fn render_len(&self) -> usize {
		self.render_cache.len()
	}

	/// 计算管线缓存中的条目数.
	pub fn compute_len(&self) -> usize {
		self.compute_cache.len()
	}

	/// 渲染管线缓存是否为空 —— 没有条目时返回 `true`.
	pub fn render_is_empty(&self) -> bool {
		self.render_cache.is_empty()
	}

	/// 计算管线缓存是否为空 —— 没有条目时返回 `true`.
	pub fn compute_is_empty(&self) -> bool {
		self.compute_cache.is_empty()
	}
}
