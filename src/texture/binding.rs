// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! `bind group` 与 `layout` 构建工具.
//!
//! wgpu 原生 API 要求手工构造 [`wgpu::BindGroupLayoutEntry`] 与
//! [`wgpu::BindGroupEntry`] 数组, 每个条目都要填 5 到 6 个字段. 对常用的
//! 「纹理 + 采样器」或「纹理 + 采样器 + uniform」组合来说, 样板代码的
//! 占比很高.
//!
//! 本模块因此提供两个 builder:
//!
//! - [`BindGroupLayoutBuilder`] —— 声明 `layout` 中的条目.
//! - [`BindGroupBuilder`] —— 绑定具体资源.
//!
//! # 使用顺序
//!
//! 先建 `layout`, 再用它建 `bind group`:
//!
//! ```ignore
//! let layout = BindGroupLayoutBuilder::new()
//!     .texture(0, wgpu::ShaderStages::FRAGMENT)
//!     .sampler(1, wgpu::ShaderStages::FRAGMENT)
//!     .uniform(2, wgpu::ShaderStages::VERTEX_FRAGMENT)
//!     .build(&device, Some("material_layout"));
//!
//! let bind_group = BindGroupBuilder::new()
//!     .texture(0, texture.view())
//!     .sampler(1, &sampler)
//!     .buffer(2, &uniform_buffer)
//!     .build(&device, &layout, Some("material"));
//! ```
//!
//! # 多次绑定与缓存
//!
//! 同一个 `layout` 可以反复创建 `bind group`. `layout` 本身应在启动阶段
//! 建好并长期持有 —— 若每次绘制都重新创建 `layout`, 管线缓存会失效.
//!
//! `bind group` 的创建成本较高 (涉及 GPU 端状态). 需要每帧绑定的资源
//! (例如每帧变化的 `uniform`) 应当提前建好 `bind group`, 每帧只调用
//! [`crate::system::PassHelper::set_bind_group`].
//!
//! # 生命周期
//!
//! [`BindGroupBuilder`] 借用资源 (纹理视图 / 采样器 / 缓冲). 这些资源在
//! builder 存续期间必须保持有效 —— 常见做法是立即调用 `.build()` 产出
//! [`wgpu::BindGroup`], 之后即可释放 builder.
//!
//! [`wgpu::BindGroup`] 自身不借用资源 —— 它持有的是 GPU 端引用. 资源
//! 释放后 `bind group` 仍然可用 (wgpu 内部维护引用计数).

use super::sampler::Sampler;

// ================ Layout 构建器 ================

/// `bind group layout` 构建器.
///
/// # 条目类型
///
/// 常用的几类绑定都有对应方法:
///
/// - [`Self::texture`] —— 2D 浮点纹理 (可过滤).
/// - [`Self::texture_with`] —— 自定义采样类型与维度.
/// - [`Self::sampler`] —— 可过滤采样器.
/// - [`Self::sampler_non_filtering`] —— 不可过滤采样器 (比较采样器等).
/// - [`Self::uniform`] —— `uniform` 缓冲.
/// - [`Self::uniform_dynamic`] —— 带 dynamic offset 的 `uniform` 缓冲.
/// - [`Self::storage_read`] / [`Self::storage_read_write`] —— `storage`
///   缓冲.
///
/// 若所需组合不在其中 (例如 3D 纹理, 存储纹理), 直接使用 wgpu 原生 API.
pub struct BindGroupLayoutBuilder {
	entries: Vec<wgpu::BindGroupLayoutEntry>,
}

impl BindGroupLayoutBuilder {
	/// 构造一个尚未包含任何条目的构建器.
	pub const fn new() -> Self {
		Self {
			entries: Vec::new(),
		}
	}

	/// 追加一个 2D 浮点纹理条目.
	///
	/// # 采样类型
	///
	/// `Float { filterable: true }` —— 支持线性过滤. 若需要精确采样 (例如
	/// 数据纹理), 改用 [`Self::texture_with`] 声明
	/// `Float { filterable: false }`.
	pub fn texture(mut self, binding: u32, visibility: wgpu::ShaderStages) -> Self {
		self.entries.push(wgpu::BindGroupLayoutEntry {
			binding,
			visibility,
			ty: wgpu::BindingType::Texture {
				sample_type: wgpu::TextureSampleType::Float { filterable: true },
				view_dimension: wgpu::TextureViewDimension::D2,
				multisampled: false,
			},
			count: None,
		});
		self
	}

	/// 追加自定义采样类型与维度的纹理条目.
	///
	/// # 参数
	///
	/// - **`binding`**: 条目在 `bind group` 中的绑定槽位编号.
	/// - **`visibility`**: 该条目可见的着色阶段.
	/// - **`sample_type`**: 采样类型 (`Float` / `Sint` / `Uint` / `Depth`).
	/// - **`view_dimension`**: 视图维度 (`D2` / `D2Array` / `Cube` 等).
	pub fn texture_with(
		mut self,
		binding: u32,
		visibility: wgpu::ShaderStages,
		sample_type: wgpu::TextureSampleType,
		view_dimension: wgpu::TextureViewDimension,
	) -> Self {
		self.entries.push(wgpu::BindGroupLayoutEntry {
			binding,
			visibility,
			ty: wgpu::BindingType::Texture {
				sample_type,
				view_dimension,
				multisampled: false,
			},
			count: None,
		});
		self
	}

	/// 追加一个可过滤采样器条目.
	///
	/// 对应 `SamplerBindingType::Filtering`, 与 [`Self::texture`] 声明的
	/// 可过滤纹理配套使用.
	pub fn sampler(mut self, binding: u32, visibility: wgpu::ShaderStages) -> Self {
		self.entries.push(wgpu::BindGroupLayoutEntry {
			binding,
			visibility,
			ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
			count: None,
		});
		self
	}

	/// 追加一个不可过滤采样器条目.
	///
	/// # 用途
	///
	/// 与 [`Self::texture_with`] 声明的 `Float { filterable: false }` 纹理
	/// 配对, 也可作为比较采样器 (阴影贴图) 使用.
	pub fn sampler_non_filtering(
		mut self,
		binding: u32,
		visibility: wgpu::ShaderStages,
	) -> Self {
		self.entries.push(wgpu::BindGroupLayoutEntry {
			binding,
			visibility,
			ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
			count: None,
		});
		self
	}

	/// 追加一个比较采样器 (`Comparison`) 条目.
	///
	/// # 用途
	///
	/// 用于阴影贴图采样. 与 [`Self::texture_with`] 声明的
	/// `TextureSampleType::Depth` 纹理配对.
	///
	/// # WGSL 声明
	///
	/// 比较采样器在着色器里声明为 `sampler_comparison`, 采样时使用
	/// `textureSampleCompare`.
	pub fn sampler_comparison(
		mut self,
		binding: u32,
		visibility: wgpu::ShaderStages,
	) -> Self {
		self.entries.push(wgpu::BindGroupLayoutEntry {
			binding,
			visibility,
			ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
			count: None,
		});
		self
	}

	/// 添加 `uniform` 缓冲条目.
	///
	/// 条目不启用 dynamic offset —— 一个 `bind group` 覆盖整段缓冲.
	pub fn uniform(mut self, binding: u32, visibility: wgpu::ShaderStages) -> Self {
		self.entries.push(wgpu::BindGroupLayoutEntry {
			binding,
			visibility,
			ty: wgpu::BindingType::Buffer {
				ty: wgpu::BufferBindingType::Uniform,
				has_dynamic_offset: false,
				min_binding_size: None,
			},
			count: None,
		});
		self
	}

	/// 添加带 dynamic offset 的 `uniform` 缓冲条目.
	///
	/// # 用途
	///
	/// 让同一个 `bind group` 按 offset 切换数据段 —— 例如每对象 `uniform`.
	/// 可参考 [`crate::system::ObjectBuffer`] 的用法.
	pub fn uniform_dynamic(mut self, binding: u32, visibility: wgpu::ShaderStages) -> Self {
		self.entries.push(wgpu::BindGroupLayoutEntry {
			binding,
			visibility,
			ty: wgpu::BindingType::Buffer {
				ty: wgpu::BufferBindingType::Uniform,
				has_dynamic_offset: true,
				min_binding_size: None,
			},
			count: None,
		});
		self
	}

	/// 添加只读 `storage` 缓冲条目.
	///
	/// 对应 `BufferBindingType::Storage { read_only: true }`, 着色器只能读取
	/// 缓冲内容.
	pub fn storage_read(mut self, binding: u32, visibility: wgpu::ShaderStages) -> Self {
		self.entries.push(wgpu::BindGroupLayoutEntry {
			binding,
			visibility,
			ty: wgpu::BindingType::Buffer {
				ty: wgpu::BufferBindingType::Storage { read_only: true },
				has_dynamic_offset: false,
				min_binding_size: None,
			},
			count: None,
		});
		self
	}

	/// 添加可读写 `storage` 缓冲条目.
	///
	/// 对应 `BufferBindingType::Storage { read_only: false }`, 着色器可以写入
	/// 缓冲内容.
	pub fn storage_read_write(
		mut self,
		binding: u32,
		visibility: wgpu::ShaderStages,
	) -> Self {
		self.entries.push(wgpu::BindGroupLayoutEntry {
			binding,
			visibility,
			ty: wgpu::BindingType::Buffer {
				ty: wgpu::BufferBindingType::Storage { read_only: false },
				has_dynamic_offset: false,
				min_binding_size: None,
			},
			count: None,
		});
		self
	}

	/// 构建 `layout`.
	///
	/// 用累积的条目调用 [`wgpu::Device::create_bind_group_layout`]. `label`
	/// 仅用于调试标记.
	pub fn build(
		self,
		device: &wgpu::Device,
		label: Option<&str>,
	) -> wgpu::BindGroupLayout {
		device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label,
			entries: &self.entries,
		})
	}
}

impl Default for BindGroupLayoutBuilder {
	fn default() -> Self {
		Self::new()
	}
}

// ================ Bind group 构建器 ================

/// `bind group` 构建器.
///
/// # 生命周期
///
/// `'a` 是所借用资源的生命周期 —— 纹理视图, 采样器, 缓冲. 这些资源
/// 在 builder 存续期间必须有效. 一般用法是立即调用 `.build()`, 不让
/// builder 长期存在.
pub struct BindGroupBuilder<'a> {
	entries: Vec<wgpu::BindGroupEntry<'a>>,
}

impl<'a> BindGroupBuilder<'a> {
	/// 构造一个尚未包含任何条目的构建器.
	pub const fn new() -> Self {
		Self {
			entries: Vec::new(),
		}
	}

	/// 把纹理视图绑定到指定槽位.
	///
	/// # 与 `layout` 的对应
	///
	/// 对应 [`BindGroupLayoutBuilder::texture`] 或
	/// [`BindGroupLayoutBuilder::texture_with`] 声明的条目. 资源类型必须与
	/// `layout` 里声明的 `sample_type` 与 `view_dimension` 一致, 否则 wgpu
	/// 会 panic.
	pub fn texture(mut self, binding: u32, view: &'a wgpu::TextureView) -> Self {
		self.entries.push(wgpu::BindGroupEntry {
			binding,
			resource: wgpu::BindingResource::TextureView(view),
		});
		self
	}

	/// 把采样器绑定到指定槽位.
	///
	/// 参数接受 [`Sampler`] 而不是 [`wgpu::Sampler`] —— `Sampler` 实现了
	/// `Deref`, 因此无需手动调用 `.raw()`.
	pub fn sampler(mut self, binding: u32, sampler: &'a Sampler) -> Self {
		self.entries.push(wgpu::BindGroupEntry {
			binding,
			resource: wgpu::BindingResource::Sampler(sampler.raw()),
		});
		self
	}

	/// 把整段缓冲绑定到指定槽位.
	///
	/// # 与 `layout` 的对应
	///
	/// 适用于不带 dynamic offset 的 `uniform` / `storage` 缓冲.
	pub fn buffer(mut self, binding: u32, buffer: &'a wgpu::Buffer) -> Self {
		self.entries.push(wgpu::BindGroupEntry {
			binding,
			resource: buffer.as_entire_binding(),
		});
		self
	}

	/// 把缓冲中的某个固定区段绑定到指定槽位.
	///
	/// # 参数
	///
	/// - **`offset`**: 起始字节偏移. 必须满足对齐要求 (`uniform` 缓冲通常要求
	///   256 字节对齐, 见 `min_uniform_buffer_offset_alignment`).
	/// - **`size`**: 段大小. 传 `None` 表示从 `offset` 一直到缓冲末尾.
	///
	/// # 与 dynamic offset 的区别
	///
	/// 这里给出的是 `bind group` 内部的**固定**偏移, 每次绘制都不变. 若要
	/// 在每次绘制时切换偏移, 改用 [`Self::buffer_dynamic`], 并在
	/// [`crate::system::PassHelper::set_bind_group`] 里传入 offset.
	pub fn buffer_slice(
		mut self,
		binding: u32,
		buffer: &'a wgpu::Buffer,
		offset: u64,
		size: Option<u64>,
	) -> Self {
		self.entries.push(wgpu::BindGroupEntry {
			binding,
			resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
				buffer,
				offset,
				size: size.and_then(wgpu::BufferSize::new),
			}),
		});
		self
	}

	/// 绑定可被 dynamic offset 选中的缓冲段.
	///
	/// # 与 `buffer_slice` 的区别
	///
	/// 这里的 `offset` 与 `size` 描述的是**单个段**的大小. 每次绘制时由
	/// [`crate::system::PassHelper::set_bind_group`] 的 `offsets` 参数挑出
	/// 实际使用的段.
	///
	/// # 前置条件
	///
	/// `layout` 里的对应条目必须由
	/// [`BindGroupLayoutBuilder::uniform_dynamic`] 声明.
	pub fn buffer_dynamic(
		mut self,
		binding: u32,
		buffer: &'a wgpu::Buffer,
		size: u64,
	) -> Self {
		self.entries.push(wgpu::BindGroupEntry {
			binding,
			resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
				buffer,
				offset: 0,
				size: wgpu::BufferSize::new(size),
			}),
		});
		self
	}

	/// 构建 `bind group`.
	///
	/// # 前置条件
	///
	/// 条目的 `binding` 编号与资源类型必须与 `layout` 一一对应. 不一致时
	/// wgpu 会在 `create_bind_group` 处 panic.
	///
	/// # 生命周期
	///
	/// 返回的 [`wgpu::BindGroup`] 不借用 `entries` 中的资源 —— 它持有 GPU 端
	/// 引用, 资源释放后仍然可用.
	pub fn build(
		self,
		device: &wgpu::Device,
		layout: &wgpu::BindGroupLayout,
		label: Option<&str>,
	) -> wgpu::BindGroup {
		device.create_bind_group(&wgpu::BindGroupDescriptor {
			label,
			layout,
			entries: &self.entries,
		})
	}
}

impl<'a> Default for BindGroupBuilder<'a> {
	fn default() -> Self {
		Self::new()
	}
}
