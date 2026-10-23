//! 顶点数据的构建管线.
//!
//! 从零拼装顶点数据, 直到产出可上传到 `wgpu` 缓冲的完整流程.
//! 本模块涵盖以下职责:
//!
//! - **格式定义**: [`VertexFormat`][] 描述顶点在缓冲中的布局, 由
//!   [`VertexFormatElement`][] 组成.
//! - **数据写入**: [`VertexConsumer`][] 定义写入协议, [`Building`][]
//!   是它的主要实现, 由 [`BufferBuilder`][] 借出.
//! - **装饰器**: [`MultiConsumer`][] 向多个下游广播;
//!   [`SheetedDecalTextureGenerator`][] 把世界坐标翻译为贴花 `UV`.
//! - **排序**: 半透明绘制要从远到近排序, 由 [`VertexSorting`][] 与
//!   它的两个实现提供.
//! - **数学辅助**: [`PoseStack`][] 维护层级变换, [`Direction`][]
//!   表示轴线.
//! - **错误**: 可能失败的操作统一返回 [`error::Error`][].
//!
//! # 典型流程
//!
//! ```
//! use blaze3d::vertex::{
//!     formats, BufferBuilder, DrawMode, VertexConsumer,
//! };
//!
//! // 1. 准备构建器 (可复用).
//! let mut builder = BufferBuilder::with_capacity(4096);
//!
//! // 2. 开始构建, 借出 Building.
//! let mut b = builder.begin(DrawMode::Quads, formats::position_color());
//!
//! // 3. 写入一个四边形 (4 个顶点).
//! for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
//!     b.vertex(x, y, 0.0)?;
//!     b.color(255, 255, 255, 255)?;
//!     b.end_vertex()?;
//! }
//!
//! // 4. 结束构建, 产出 RenderedBuffer.
//! let rendered = b.end();
//!
//! // 5. 上传到 GPU.
//! // let uploaded = rendered.upload(&device, &queue);
//! # Ok::<(), blaze3d::vertex::Error>(())
//! ```
//!
//! # 数据流
//!
//! 一次完整构建依次经历以下阶段:
//!
//! 1. `VertexFormat::new` 校验格式并生成 `wgpu` 属性列表.
//! 2. `BufferBuilder::begin` 借出 `Building`, 绑定图元模式与格式.
//! 3. 通过 `VertexConsumer` 的方法逐个写入顶点, 由 `Building` 按
//!    格式顺序推进.
//! 4. `Building::end` 产出 `RenderedBuffer`, 内含顶点字节与 (可选的)
//!    索引字节.
//! 5. `RenderedBuffer::upload` 创建 `wgpu` 缓冲并写入数据, 产出
//!    `UploadedBuffer`.
//!
//! # 状态约束
//!
//! `BufferBuilder` 与 `Building` 之间的借用关系在编译期约束构建状态:
//!
//! - `begin` 要求 `&mut BufferBuilder`, 返回持有该借用的 `Building`.
//! - `Building` 存活期间无法再次 `begin` —— 借用检查器会拒绝.
//! - 写入方法只能通过 `Building` 调用, 「未开始就写入」没有对应
//!   `API`.
//!
//! 因此这两类状态机错误无需运行时检查, 也不会产生 panic.

// ================ 子模块声明 ================

pub mod builder;
pub mod consumer;
pub mod decal;
pub mod direction;
pub mod element;
pub mod error;
pub mod format;
pub mod pose;
pub mod sorting;

// ================ 公开重导出 ================
//
// 仅导出最常用的类型与函数. 需要完整路径时, 仍可经子模块访问.

// -- element: 元素类型与用法 --

pub use element::{ElementType, ElementUsage, VertexFormatElement};

// -- format: 顶点格式与内置格式 --

pub use format::defaults as formats;
pub use format::{SharedVertexFormat, VertexFormat};

// -- consumer: 写入协议与广播装饰器 --

pub use consumer::{MultiConsumer, VertexConsumer};

// -- builder: 构建器与产物 --

pub use builder::{
	BufferBuilder, Building, DrawMode, DrawState, IndexType, RenderedBuffer, UploadedBuffer,
};

// -- 装饰器: 贴花 `UV` 生成 --

pub use decal::SheetedDecalTextureGenerator;

// -- 数学辅助: 朝向与位姿 --

pub use direction::Direction;
pub use pose::{Pose, PoseGuard, PoseStack};

// -- sorting: 顶点排序策略 --

pub use sorting::{DistanceSorting, OrthographicZSorting, VertexSorting};

// -- error: 错误类型与结果别名 --

pub use error::{Error, Result};
