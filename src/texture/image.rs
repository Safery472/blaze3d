// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 纹理资源的封装.
//!
//! [`Texture`] 表示一张 GPU 纹理, 支持三种维度:
//!
//! - [`TextureKind::D2`] —— 2D 纹理, 最常见的一类.
//! - [`TextureKind::D3`] —— 3D 体素纹理, 承载体积数据 (流体模拟结果, 密度场,
//!   3D 噪声).
//! - [`TextureKind::Cube`] —— 立方体贴图, 由 6 个面构成, 用于天空盒与环境
//!   映射.
//!
//! # 与 `pipeline::RenderTarget` 的区别
//!
//! [`crate::pipeline::RenderTarget`] 是**渲染目标** —— 专供 render pass 输出,
//! 内部自行管理颜色附件与深度附件. [`Texture`] 则是**通用纹理** —— 既能当
//! 采样源, 也能借 `RENDER_ATTACHMENT` 用途充当渲染目标, 但不负责封装附件
//! 逻辑.
//!
//! 既要「渲染到纹理」又要回采样时选 `RenderTarget`; 只需要一张纹理给 shaders
//! 当输入时选 [`Texture`].
//!
//! # Mipmap
//!
//! [`Texture::from_bytes`] 只接受 `mip_level_count == 1` —— 上传多 Mip 数据
//! 需要知道每一层的独立尺寸, API 会随之变复杂. 需要多 Mip 时先用
//! [`Texture::new`] 建一张空纹理, 再逐层 `queue.write_texture`.
//!
//! [`Texture::generate_mipmaps`] 覆盖 `D2` / `D3` / `Cube` 三种维度, 各自使用
//! 配套的 compute shaders, 详见 [`crate::texture::mipmap`] 的模块文档.
//!
//! # 用途
//!
//! 纹理能做什么由 `usage` 决定:
//!
//! - `TEXTURE_BINDING`: 可以充当 bind group 里的纹理资源.
//! - `RENDER_ATTACHMENT`: 可以充当 render pass 的颜色附件.
//! - `STORAGE_BINDING`: 可以作为 compute shaders 的 storage texture.
//! - `COPY_DST`: 允许从 CPU 上传.
//! - `COPY_SRC`: 允许被复制到其他资源.
//!
//! `from_bytes` 给出的默认 `usage` 是 `TEXTURE_BINDING | COPY_DST` —— 既可
//! 采样也可上传. 还需要别的用途时, 用 [`TextureDesc::with_usage`] 显式声明.
//!
//! # Cube 面顺序
//!
//! `from_bytes` 上传 Cube 纹理时, 数据必须按 wgpu / WebGPU 约定的顺序摆放:
//!
//! ```text
//! 层 0: +X
//! 层 1: -X
//! 层 2: +Y
//! 层 3: -Y
//! 层 4: +Z
//! 层 5: -Z
//! ```
//!
//! 各层首尾相接, 层与层之间不留 padding.

use super::error::{Error, Result};

// ================ 维度 ================

/// 纹理维度的种类.
///
/// 它决定 [`Texture`] 在 GPU 上的实际维度, 也决定默认视图如何解释这张纹理.
///
/// # 与 `wgpu::TextureDimension` 的差异
///
/// `wgpu` 只有 `D1` / `D2` / `D3` 三种维度; 立方体贴图在那一层看来就是
/// 「6 层的 `D2` 数组」, 于是视图维度 (`Cube`) 与纹理维度 (`D2`) 成了两个
/// 独立概念.
///
/// 本枚举把「纹理维度 + 视图意图」合成一个直观选项 —— 对使用者而言「立方体
/// 贴图」是一个整体概念, 而不是「6 层 D2 加一个 Cube 视图」.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureKind {
	/// 二维 (2D) 纹理.
	D2,
	/// 三维体素纹理.
	///
	/// 第三维的尺寸由 [`TextureDesc::depth`] 给出.
	D3,
	/// 立方体贴图, 由 6 个面组成.
	///
	/// 尺寸恒为正方形 —— [`TextureDesc::width`] 与 [`TextureDesc::height`] 必须
	/// 相等.
	///
	/// 默认视图的维度是 [`wgpu::TextureViewDimension::Cube`], 因此可以直接用于
	/// `texture_cube` 采样.
	Cube,
}

// ================ 描述 ================

/// 纹理的创建描述.
///
/// # 构造
///
/// 先用 [`Self::new_2d`] / [`Self::new_3d`] / [`Self::new_cube`] 之一建出描述,
/// 再用链式方法覆盖不需要的默认值:
///
/// ```ignore
/// let desc = TextureDesc::new_2d(256, 256, wgpu::TextureFormat::Rgba8UnormSrgb)
///     .with_label("albedo")
///     .with_usage(wgpu::TextureUsages::TEXTURE_BINDING
///         | wgpu::TextureUsages::RENDER_ATTACHMENT
///         | wgpu::TextureUsages::COPY_DST);
/// ```
///
/// # 字段
///
/// 字段全部公开, 直接写 `TextureDesc` 字面量同样合法; 不过更推荐走构造函数
/// 加 builder, 它们会填好合理的默认值.
#[derive(Debug, Clone)]
pub struct TextureDesc<'a> {
	/// 调试时显示的标签.
	pub label: Option<&'a str>,

	/// 决定纹理在 GPU 上的维度.
	pub kind: TextureKind,

	/// 纹理宽度.
	///
	/// 对 `Cube` 而言即单个面的宽度, 此时与 `height` 相等.
	pub width: u32,

	/// 纹理高度.
	///
	/// 对 `Cube` 而言与 `width` 相等.
	pub height: u32,

	/// 纹理深度.
	///
	/// 仅 [`TextureKind::D3`] 会用到; `D2` 与 `Cube` 忽略该字段.
	pub depth: u32,

	/// 纹理的像素格式.
	pub format: wgpu::TextureFormat,

	/// 允许的用途, 以位标志表示.
	pub usage: wgpu::TextureUsages,

	/// Mipmap 层数, 取 1 表示不带 Mip.
	pub mip_level_count: u32,

	/// 采样数, 取 1 表示非 MSAA.
	///
	/// **MSAA 仅支持 `D2`** —— `D3` 与 `Cube` 传入大于 1 的值时,
	/// [`Texture::new`] 会返回 [`Error::IncompatibleKind`].
	pub sample_count: u32,
}

impl<'a> TextureDesc<'a> {
	/// 构造一条 2D 纹理描述.
	///
	/// # 默认值
	///
	/// - `usage`: `TEXTURE_BINDING | COPY_DST`, 即既能采样也能上传.
	/// - `mip_level_count`: 1.
	/// - `sample_count`: 1.
	pub fn new_2d(width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
		Self {
			label: None,
			kind: TextureKind::D2,
			width,
			height,
			depth: 1,
			format,
			usage: wgpu::TextureUsages::TEXTURE_BINDING
				| wgpu::TextureUsages::COPY_DST,
			mip_level_count: 1,
			sample_count: 1,
		}
	}

	/// 构造一条 3D 纹理描述.
	///
	/// # 参数
	///
	/// - `depth`: 第三维的尺寸, 不能为 0.
	///
	/// # 默认值
	///
	/// 除 `kind` 与 `depth` 外, 其余字段与 [`Self::new_2d`] 一致.
	pub fn new_3d(
		width: u32,
		height: u32,
		depth: u32,
		format: wgpu::TextureFormat,
	) -> Self {
		Self {
			label: None,
			kind: TextureKind::D3,
			width,
			height,
			depth,
			format,
			usage: wgpu::TextureUsages::TEXTURE_BINDING
				| wgpu::TextureUsages::COPY_DST,
			mip_level_count: 1,
			sample_count: 1,
		}
	}

	/// 构造一条立方体贴图描述.
	///
	/// # 参数
	///
	/// - `size`: 单个面的边长, 6 个面都是 `size x size`.
	///
	/// # 默认值
	///
	/// 其余字段与 [`Self::new_2d`] 相同, 默认视图则是 Cube 视图.
	pub fn new_cube(size: u32, format: wgpu::TextureFormat) -> Self {
		Self {
			label: None,
			kind: TextureKind::Cube,
			width: size,
			height: size,
			depth: 1,
			format,
			usage: wgpu::TextureUsages::TEXTURE_BINDING
				| wgpu::TextureUsages::COPY_DST,
			mip_level_count: 1,
			sample_count: 1,
		}
	}

	/// 覆盖调试标签.
	pub fn with_label(mut self, label: &'a str) -> Self {
		self.label = Some(label);
		self
	}

	/// 覆盖纹理用途.
	///
	/// **注意**: 这里是整体覆盖, 不是追加. 想要多个用途, 必须自己用 `|` 组合后
	/// 传入.
	pub fn with_usage(mut self, usage: wgpu::TextureUsages) -> Self {
		self.usage = usage;
		self
	}

	/// 覆盖 Mipmap 层数.
	pub fn with_mip_levels(mut self, count: u32) -> Self {
		self.mip_level_count = count;
		self
	}

	/// 覆盖采样数.
	///
	/// **MSAA 仅支持 `D2`**; `D3` / `Cube` 传入大于 1 的值会在创建纹理时报错.
	pub fn with_sample_count(mut self, count: u32) -> Self {
		self.sample_count = count;
		self
	}

	/// 层数 —— `D3` 取深度值, `Cube` 取 6 个面.
	///
	/// 计算数据大小, 以及填 `write_texture` 的 `depth_or_array_layers` 时都要
	/// 用到它.
	pub const fn layers(&self) -> u32 {
		match self.kind {
			TextureKind::D2 => 1,
			TextureKind::D3 => self.depth,
			TextureKind::Cube => 6,
		}
	}

	/// 转换成 wgpu 的纹理描述符.
	pub(crate) fn to_wgpu(&self) -> wgpu::TextureDescriptor<'_> {
		let (dimension, depth_or_array_layers) = match self.kind {
			TextureKind::D2 => (wgpu::TextureDimension::D2, 1),
			TextureKind::D3 => (wgpu::TextureDimension::D3, self.depth),
			TextureKind::Cube => (wgpu::TextureDimension::D2, 6),
		};

		wgpu::TextureDescriptor {
			label: self.label,
			size: wgpu::Extent3d {
				width: self.width,
				height: self.height,
				depth_or_array_layers,
			},
			mip_level_count: self.mip_level_count,
			sample_count: self.sample_count,
			dimension,
			format: self.format,
			usage: self.usage,
			view_formats: &[],
		}
	}
}

// ================ 纹理 ================

/// 位于 GPU 上的纹理.
///
/// # 视图
///
/// 创建纹理时会一并生成默认视图, 它覆盖全部 Mip 与全部层, 由 [`Self::view`]
/// 取出. 视图维度取决于 [`TextureKind`]:
///
/// - `D2`: 默认视图维度为 [`wgpu::TextureViewDimension::D2`].
/// - `D3`: 默认视图维度为 [`wgpu::TextureViewDimension::D3`].
/// - `Cube`: 默认视图维度为 [`wgpu::TextureViewDimension::Cube`].
///
/// 若要单个 Mip 或单个层的视图, 改用 [`Self::create_view`].
///
/// # 生命周期
///
/// 底层 [`wgpu::Texture`] 与 [`wgpu::TextureView`] 都随 `Texture` 一同 drop,
/// 无需手工释放.
pub struct Texture {
	texture: wgpu::Texture,
	view: wgpu::TextureView,

	kind: TextureKind,
	width: u32,
	height: u32,
	depth: u32,
	format: wgpu::TextureFormat,
	mip_level_count: u32,
	sample_count: u32,
}

impl Texture {
	/// 创建一张空纹理.
	///
	/// # 行为
	///
	/// - 校验尺寸非零, 且不超出设备上限.
	/// - 校验维度与其余参数相容 (MSAA 只能配 `D2`).
	/// - 创建 [`wgpu::Texture`] 并生成默认视图.
	/// - 不填充内容 —— 作为渲染目标或 storage 纹理时, 数据由 GPU 写入.
	///
	/// # 错误
	///
	/// - 任一维尺寸为 0: [`Error::ZeroSize`].
	/// - 尺寸越过设备上限: [`Error::SizeExceedsLimit`].
	/// - 维度与参数不相容: [`Error::IncompatibleKind`].
	pub fn new(device: &wgpu::Device, desc: &TextureDesc<'_>) -> Result<Self> {
		validate_desc(device, desc)?;

		let texture = device.create_texture(&desc.to_wgpu());
		let view = create_default_view(&texture, desc.kind);

		Ok(Self {
			texture,
			view,
			kind: desc.kind,
			width: desc.width,
			height: desc.height,
			depth: desc.depth,
			format: desc.format,
			mip_level_count: desc.mip_level_count,
			sample_count: desc.sample_count,
		})
	}

	/// 用一段字节数据创建纹理.
	///
	/// # 数据布局
	///
	/// 数据按行主序 (row-major) 排列, 层与层紧密相接:
	///
	/// - `D2`: 共 `height` 行, 每行 `width * block_size` 字节.
	/// - `D3`: 共 `depth` 层, 每层 `height` 行.
	/// - `Cube`: 共 6 层, 每层 `height` 行.
	///
	/// **不要求** 256 字节对齐 —— 对齐由 `write_texture` 内部搞定.
	///
	/// # Cube 面顺序
	///
	/// 参见模块文档中的「Cube 面顺序」一节.
	///
	/// # 错误
	///
	/// - 任一维尺寸为 0: [`Error::ZeroSize`].
	/// - 尺寸越过设备上限: [`Error::SizeExceedsLimit`].
	/// - 维度与参数不相容: [`Error::IncompatibleKind`].
	/// - 格式无法从 CPU 上传: [`Error::UnsupportedFormat`].
	/// - 数据大小与纹理尺寸不符: [`Error::DataSizeMismatch`].
	/// - 请求了多级 Mip: [`Error::UnsupportedMipLevels`].
	/// - 请求了 MSAA 采样: [`Error::UnsupportedSampleCount`].
	pub fn from_bytes(
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		desc: &TextureDesc<'_>,
		data: &[u8],
	) -> Result<Self> {
		validate_desc(device, desc)?;

		if desc.mip_level_count != 1 {
			return Err(Error::UnsupportedMipLevels {
				requested: desc.mip_level_count,
			});
		}

		// MSAA 纹理只能当 render pass 附件, 无法接收 CPU 端上传的数据. 这里用
		// 独立变体与 Mip 层数问题区分开, 调用方更容易定位原因.
		if desc.sample_count != 1 {
			return Err(Error::UnsupportedSampleCount {
				requested: desc.sample_count,
			});
		}

		let block_size = desc
			.format
			.block_copy_size(None)
			.ok_or(Error::UnsupportedFormat { format: desc.format })?;

		// 期望的字节数 = 宽 * 高 * 层数 * 每像素字节数.
		let expected = desc.width as u64
			* desc.height as u64
			* desc.layers() as u64
			* block_size as u64;
		if data.len() as u64 != expected {
			return Err(Error::DataSizeMismatch {
				expected,
				actual: data.len() as u64,
			});
		}

		let texture = device.create_texture(&desc.to_wgpu());
		upload_bytes(queue, &texture, desc, data, block_size);

		let view = create_default_view(&texture, desc.kind);

		Ok(Self {
			texture,
			view,
			kind: desc.kind,
			width: desc.width,
			height: desc.height,
			depth: desc.depth,
			format: desc.format,
			mip_level_count: desc.mip_level_count,
			sample_count: desc.sample_count,
		})
	}

	// ================ 访问器 ================

	/// 底层的 wgpu 纹理对象.
	pub fn raw(&self) -> &wgpu::Texture {
		&self.texture
	}

	/// 覆盖全部 Mip 与全部层的默认视图.
	///
	/// 视图维度取决于 [`TextureKind`], 详见类型文档.
	pub fn view(&self) -> &wgpu::TextureView {
		&self.view
	}

	/// 创建时选定的维度.
	pub fn kind(&self) -> TextureKind {
		self.kind
	}

	/// 纹理宽度 (像素).
	pub fn width(&self) -> u32 {
		self.width
	}

	/// 纹理高度 (像素).
	pub fn height(&self) -> u32 {
		self.height
	}

	/// 纹理深度.
	///
	/// `D2` 与 `Cube` 恒为 1; `D3` 取创建时传入的 `depth`.
	pub fn depth(&self) -> u32 {
		self.depth
	}

	/// 像素格式.
	pub fn format(&self) -> wgpu::TextureFormat {
		self.format
	}

	/// 这张纹理的 Mipmap 层数.
	pub fn mip_level_count(&self) -> u32 {
		self.mip_level_count
	}

	/// 这张纹理的采样数.
	pub fn sample_count(&self) -> u32 {
		self.sample_count
	}

	/// 以 `(width, height)` 形式返回尺寸.
	///
	/// 3D 纹理的深度不在这里, 需通过 [`Self::depth`] 单独访问.
	pub fn size(&self) -> (u32, u32) {
		(self.width, self.height)
	}

	// ================ 视图 ================

	/// 自行指定参数创建视图.
	///
	/// # 用途
	///
	/// - 只取某个 Mip 层 (`base_mip_level` 与 `mip_level_count`).
	/// - 只取某些数组层 (`base_array_layer` 与 `array_layer_count`).
	/// - 换一种维度解释 —— 例如把 Cube 纹理的某个面当成 `D2` 来采样.
	///
	/// # 建议
	///
	/// 视图若要放进 bind group, 创建后应当一直持有 —— 每次绘制都新建视图会造成
	/// GPU 资源泄漏.
	pub fn create_view(&self, desc: &wgpu::TextureViewDescriptor<'_>) -> wgpu::TextureView {
		self.texture.create_view(desc)
	}
}

// ================ 内部辅助 ================

/// 检查描述里的各项是否合法.
///
/// # 检查项
///
/// - 宽, 高与深度都不能为 0.
/// - 各维尺寸不得超出设备对应维度的上限.
/// - MSAA 只能配 `D2`.
/// - `Cube` 的宽高必须相等.
fn validate_desc(device: &wgpu::Device, desc: &TextureDesc<'_>) -> Result<()> {
	if desc.width == 0 || desc.height == 0 {
		return Err(Error::ZeroSize {
			width: desc.width,
			height: desc.height,
			depth: desc.depth,
		});
	}
	if desc.kind == TextureKind::D3 && desc.depth == 0 {
		return Err(Error::ZeroSize {
			width: desc.width,
			height: desc.height,
			depth: desc.depth,
		});
	}

	// Cube 的 6 个面尺寸必须一致 —— wgpu 的 Cube 视图要求贴图为正方形.
	if desc.kind == TextureKind::Cube && desc.width != desc.height {
		return Err(Error::IncompatibleKind {
			kind: desc.kind,
			reason: "cube texture must be square (width == height)",
		});
	}

	let limits = device.limits();

	// 上限的取法: D3 查 3D 上限, D2 与 Cube 查 2D 上限.
	let limit = match desc.kind {
		TextureKind::D3 => limits.max_texture_dimension_3d,
		TextureKind::D2 | TextureKind::Cube => limits.max_texture_dimension_2d,
	};

	if desc.width > limit || desc.height > limit {
		return Err(Error::SizeExceedsLimit {
			width: desc.width,
			height: desc.height,
			depth: desc.depth,
			limit,
		});
	}
	if desc.kind == TextureKind::D3 && desc.depth > limits.max_texture_dimension_3d {
		return Err(Error::SizeExceedsLimit {
			width: desc.width,
			height: desc.height,
			depth: desc.depth,
			limit,
		});
	}

	// MSAA 仅能用在 2D 纹理上.
	if desc.sample_count > 1 && desc.kind != TextureKind::D2 {
		return Err(Error::IncompatibleKind {
			kind: desc.kind,
			reason: "MSAA is only supported for D2 textures",
		});
	}

	Ok(())
}

/// 按 kind 生成默认视图.
fn create_default_view(texture: &wgpu::Texture, kind: TextureKind) -> wgpu::TextureView {
	match kind {
		// D2 与 D3 的默认视图维度与纹理维度一致, 直接用 Default 即可.
		TextureKind::D2 | TextureKind::D3 => {
			texture.create_view(&wgpu::TextureViewDescriptor::default())
		}
		// Cube 纹理在 wgpu 层面是 6 层的 D2 数组, 默认视图必须显式声明维度 ——
		// 取 `D2Array` 或 `Cube` 二者之一.
		//
		// **这里选 `Cube` 而不是 `D2Array`**: 创建 Cube 纹理的意图就是
		// 在 shaders 里用 `texture_cube` 采样, 而 `Cube` 视图维度正好
		// 与该采样方式匹配.
		TextureKind::Cube => {
			texture.create_view(&wgpu::TextureViewDescriptor {
				label: Some("texture_cube_view"),
				dimension: Some(wgpu::TextureViewDimension::Cube),
				..Default::default()
			})
		}
	}
}

/// 把字节数据上传进纹理.
///
/// 三种 kind 都只需一次 `write_texture`: 靠 `depth_or_array_layers` 让 wgpu
/// 一次性写完所有层.
fn upload_bytes(
	queue: &wgpu::Queue,
	texture: &wgpu::Texture,
	desc: &TextureDesc<'_>,
	data: &[u8],
	block_size: u32,
) {
	let bytes_per_row = desc.width * block_size;

	queue.write_texture(
		wgpu::TexelCopyTextureInfo {
			texture,
			mip_level: 0,
			origin: wgpu::Origin3d::ZERO,
			aspect: wgpu::TextureAspect::All,
		},
		data,
		wgpu::TexelCopyBufferLayout {
			offset: 0,
			bytes_per_row: Some(bytes_per_row),
			rows_per_image: Some(desc.height),
		},
		wgpu::Extent3d {
			width: desc.width,
			height: desc.height,
			depth_or_array_layers: desc.layers(),
		},
	);
}
