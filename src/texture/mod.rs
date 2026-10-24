// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 纹理资源与绑定资源的封装.
//!
//! 本模块提供四类基础设施, 分工如下:
//!
//! - [`Texture`] —— 纹理资源, 支持 2D / 3D / 立方体贴图三种维度.
//! - [`Sampler`] —— 采样器, 内置最近邻 / 双线性 / 三线性预设.
//! - [`MipmapGenerator`] —— Mipmap 生成器, 优先走 `compute`, 回退到
//!   `render` 路径.
//! - [`BindGroupLayoutBuilder`] / [`BindGroupBuilder`] —— `bind group` 与
//!   `layout` 的构建工具, 用于简化 `wgpu` 中繁琐的条目声明.
//!
//! # 与相邻模块的关系
//!
//! - 顶点格式与写入: 由 [`crate::vertex`] 定义.
//! - 渲染目标: 由 [`crate::pipeline`] 定义.
//! - 长期渲染资源与逐帧驱动: 由 [`crate::system`] 负责.
//! - 绘制输入: 由 `texture` 提供 —— 纹理与采样器经 `bind group` 绑定到
//!   管线上, 供 `shaders` 采样.
//!
//! # 典型流程
//!
//! 1. 用解码后的图像字节创建 [`Texture`].
//! 2. 创建 [`Sampler`], 并选定过滤模式.
//! 3. 用 [`BindGroupLayoutBuilder`] 声明 `layout`.
//! 4. 用 [`BindGroupBuilder`] 绑定具体的纹理与采样器.
//! 5. 在 [`crate::system::PassHelper::set_bind_group`] 中完成绑定.
//!
//! ```ignore
//! let texture = Texture::from_bytes(
//!     &device, &queue,
//!     &TextureDesc::new_2d(256, 256, wgpu::TextureFormat::Rgba8UnormSrgb),
//!     &rgba_data,
//! )?;
//! let sampler = Sampler::linear(&device, Some("linear"));
//!
//! let layout = BindGroupLayoutBuilder::new()
//!     .texture(0, wgpu::ShaderStages::FRAGMENT)
//!     .sampler(1, wgpu::ShaderStages::FRAGMENT)
//!     .build(&device, Some("material_layout"));
//!
//! let bind_group = BindGroupBuilder::new()
//!     .texture(0, texture.view())
//!     .sampler(1, &sampler)
//!     .build(&device, &layout, Some("material"));
//! ```
//!
//! # 维度
//!
//! [`TextureKind`] 共区分三种形态:
//!
//! - `D2` —— 普通 2D 纹理, 默认视图维度为 `D2`.
//! - `D3` —— 3D 体素纹理, 第三维由 [`TextureDesc::depth`] 指定, 默认
//!   视图维度为 `D3`.
//! - `Cube` —— 立方体贴图, 共 6 个面, 尺寸恒为正方形, 默认视图维度
//!   为 `Cube`, 可直接用 `texture_cube` 采样.
//!
//! 为 Cube 绑定 `bind group` 时, `BindGroupLayoutBuilder` 需要用
//! [`BindGroupLayoutBuilder::texture_with`] 声明 `Cube` 维度 —— 默认的
//! `texture` 方法声明的是 `D2`.
//!
//! # Mipmap
//!
//! [`Texture::generate_mipmaps`] 通过 `compute` (优先) 或 `render pass`
//! (回退) 生成 Mipmap, 支持 `D2` / `D3` / `Cube` —— 三种维度各自使用
//! 对应的 `compute shaders`. `render` 回退路径只对 `D2` 有定义.
//!
//! 需要多次生成时, 用 [`MipmapGenerator`] 做缓存 —— 这样构造开销与
//! 首次针对某种格式的管线编译开销都能省下.
//!
//! # 子模块命名
//!
//! `Texture` 所在的文件命名为 `image.rs` 而不是 `texture.rs` —— 若叫
//! 后者, 模块路径会变成 `texture::texture`, 触发 `Clippy` 的
//! `module_inception` 警告. `image` 这个名字表达了「纹理源自图像
//! 数据」的语义.

// ================ 子模块声明 ================

pub mod binding;
pub mod error;
pub mod image;
pub mod mipmap;
pub mod sampler;

// ================ 对外重导出 ================

pub use binding::{BindGroupBuilder, BindGroupLayoutBuilder};
pub use error::{Error, Result};
pub use image::{Texture, TextureDesc, TextureKind};
pub use mipmap::MipmapGenerator;
pub use sampler::Sampler;
