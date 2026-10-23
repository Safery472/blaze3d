//! 渲染管线.
//!
//! 本模块是「渲染目标」到「帧提交」之间的中间层. 与 [`crate::vertex`] 的
//! 分工如下:
//!
//! - `vertex` 负责**数据**: 顶点格式, 顶点写入, 缓冲上传.
//! - `pipeline` 负责**目标**: 渲染到哪里 (纹理 / 窗口), 使用什么尺寸,
//!   清屏参数, 深度附件, MSAA.
//!
//! # 与 Java 版 `pipeline` 包的差异
//!
//! Java 版有 5 个类: `RenderCall` / `RenderPipeline` / `RenderTarget` /
//! `MainTarget` / `TextureTarget`. 在 wgpu 下只有 [`RenderTarget`] 这一
//! 部分需要保留, 其余四者的职责被其他机制覆盖或合并:
//!
//! - **`RenderCall` / `RenderPipeline`**: 旧版为解决「渲染必须在单一
//!   线程执行」而引入的命令队列. wgpu 的 [`wgpu::Queue`] 与
//!   [`wgpu::Device`] 都实现了 `Send + Sync`, 不需要队列机制.
//! - **`MainTarget`**: 与 `RenderTarget` 的唯一区别是尺寸超限时的回退
//!   逻辑, 用 [`clamp_to_limits`] 在构造前处理即可.
//! - **`TextureTarget`**: 与 `RenderTarget` 完全没有差异.
//!
//! # 窗口集成
//!
//! wgpu 的窗口渲染通过 [`wgpu::Surface`] 进行. [`SurfaceFrame`] 把一次
//! surface 纹理获取包装成实现了 [`Target`] 的类型. [`configure_surface`]
//! 负责格式协商与 present 模式选择, [`reconfigure_surface`] 处理运行时
//! 的尺寸变化.
//!
//! # 深度与 MSAA
//!
//! - [`RenderTarget`] 内部按 `depth_format` 与 `sample_count` 参数创建
//!   深度纹理与 MSAA 纹理.
//! - [`SurfaceFrame`] 不携带附件 —— 深度与 MSAA 由 [`DepthBuffer`] 和
//!   [`MsaaBuffer`] 单独提供, 传给 [`SurfaceFrame::acquire`].
//!
//! MSAA 启用时, 深度缓冲与 MSAA 缓冲的 `sample_count` 必须一致, 这一点
//! 由调用方保证.
//!
//! # 阴影贴图
//!
//! - [`ShadowMap`] 用于**方向光**阴影: 从光源视角 (正交投影) 渲染深度,
//!   主 pass 用比较采样器采样.
//! - [`CubeShadowMap`] 用于**点光 / 聚光**阴影: 立方体深度纹理, 6 个面
//!   各覆盖 90°, 主 pass 用方向向量采样.
//!
//! # 读回
//!
//! [`read_texture`] / [`read_target`] / [`read_buffer`] 把 GPU 上的资源
//! 复制回 CPU.
//!
//! # wgpu 30
//!
//! - `SurfaceTexture::present()` 已移除, 改用 [`wgpu::Queue::present`].
//! - [`wgpu::SurfaceConfiguration`] 新增 `color_space` 字段.
//! - [`wgpu::RequestAdapterOptions`] 新增 `apply_limit_buckets` 字段.

// ================ 子模块 ================

pub mod depth;
pub mod error;
pub mod msaa;
pub mod readback;
pub mod shadow;
pub mod cube;
pub mod surface;
pub mod target;

// ================ 重导出 ================

pub use depth::DepthBuffer;
pub use error::{Error, Result};
pub use msaa::MsaaBuffer;
pub use readback::{read_buffer, read_target, read_texture};
pub use shadow::ShadowMap;
pub use cube::{CubeFace, CubeShadowMap};
pub use surface::{configure_surface, reconfigure_surface, SurfaceFrame};
pub use target::{clamp_to_limits, RenderTarget, Target, DEFAULT_HEIGHT, DEFAULT_WIDTH};
