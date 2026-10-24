// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 管线状态描述与布局句柄.
//!
//! 本模块导出以下内容:
//!
//! - [`LayoutHandle`] —— 管线布局句柄, 把 wgpu 布局与唯一 `id` 绑定在一起.
//! - [`RenderType`] / [`RenderTypeBuilder`] —— 描述一次绘制的完整管线状态.
//! - [`PipelineCache`] —— 依 [`PipelineDesc`] / [`ComputePipelineDesc`]
//!   复用已经建好的管线.

pub mod layout;
pub mod pipeline;

pub use layout::LayoutHandle;
pub use pipeline::{
	BlendMode, ColorFormat, CompareOp, ComputePipelineDesc, CullMode, DepthFormat,
	DepthMode, PipelineCache, PipelineDesc, PolygonMode, RenderType, RenderTypeBuilder,
	WriteMask,
};
