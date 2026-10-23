//! 后处理.
//!
//! 渲染完成后对整帧做图像处理. 典型流程:
//!
//! 1. 把场景渲染到离屏的 [`crate::pipeline::RenderTarget`], 而不是直接
//!    渲染到窗口.
//! 2. [`PostProcessChain`] 用 [`FullscreenQuad`] 采样场景纹理, 应用效果,
//!    再输出到目标 (通常是窗口).
//!
//! # 模块结构
//!
//! - [`FullscreenQuad`]: 全屏四边形, 后处理的几何输入.
//! - [`PostProcessChain`]: 后处理链, 持有全屏四边形并管理多个 pass.
//! - [`PostProcessPass`]: 单个 pass, 由着色器, 布局与目标组成.
//! - [`PostProcessInput`]: pass 的输入来源.
//!
//! # 多 pass
//!
//! 链支持任意数量的 pass. 每个 pass 的输入可以是外部 bind group, 也可以
//! 是上一个 pass 的输出. 中间的 pass 输出到链内部的中间目标池, 池按
//! ping-pong 策略复用 2 个目标.
//!
//! # 为什么不用全屏三角形
//!
//! 全屏三角形 (用一个覆盖全屏的三角形代替四边形) 的光栅化效率更高:
//! 没有对角线上的重复像素. 但对后处理来说四边形更直观 —— UV 与屏幕空间
//! 一一对应, 也不会出现三角形顶点落在屏幕外的特殊情况.
//!
//! 本模块使用四边形. 需要三角形时, 替换 [`FullscreenQuad::new`] 的实现
//! 即可.
//!
//! # 线性链 vs 图
//!
//! - [`PostProcessChain`] 是**线性**的: 每个 pass 的输入是外部纹理或上一
//!   个 pass, 覆盖大多数需求, API 也简单.
//! - [`PostProcessGraph`] 是**有向无环图**: 支持分支 (一个输出被多个 pass
//!   读取) 与合流 (一个 pass 读取多个输入). 需要泛光 + 景深这类结构时
//!   应当用图.
//!
//! # 示例
//!
//! 单 pass 场景 (场景纹理 → 色调映射 → 窗口):
//!
//! ```ignore
//! // 启动阶段.
//! let scene_texture_layout = BindGroupLayoutBuilder::new()
//!     .texture(0, wgpu::ShaderStages::FRAGMENT)
//!     .sampler(1, wgpu::ShaderStages::FRAGMENT)
//!     .build(&device, Some("scene_texture_layout"));
//! let scene_sampler = Sampler::linear(&device, None);
//!
//! let mut chain = PostProcessChain::new(
//!     &device, &queue,
//!     scene_texture_layout.clone(),
//!     scene_sampler.raw(),
//! );
//!
//! // 每个 pass 的 layout: group 0 = globals, group 1 = 输入纹理.
//! let tonemap_globals_layout = system.globals_bind_group_layout().clone();
//! let tonemap_layout = LayoutHandle::new(
//!     &device,
//!     &[&tonemap_globals_layout, &scene_texture_layout],
//! );
//! let tonemap_rt = Arc::new(fullscreen_render_type(window_format, "tonemap"));
//! let tonemap_shader = /* ... */;
//!
//! chain.add_pass(
//!     tonemap_rt, tonemap_shader, tonemap_layout,
//!     PostProcessInput::External,
//! );
//!
//! // 场景纹理的 bind group.
//! let scene_bind_group = BindGroupBuilder::new()
//!     .texture(0, scene_target.color_view())
//!     .sampler(1, &scene_sampler)
//!     .build(&device, &scene_texture_layout, Some("scene_texture"));
//!
//! // 每帧.
//! let mut frame = system.begin_frame();
//!
//! // 场景 → 离屏目标.
//! frame.pass(&scene_target, |pass| { /* 场景绘制 */ });
//!
//! // 后处理链 → 窗口.
//! let surface_frame = SurfaceFrame::acquire(...)?;
//! chain.draw_into_frame(
//!     &mut frame,
//!     &surface_frame,
//!     &scene_bind_group,
//!     &device,
//! )?;
//!
//! frame.submit();
//! surface_frame.present(&queue);
//! ```

pub mod chain;
pub mod graph;
pub mod quad;

pub use chain::{PostProcessChain, PostProcessInput, PostProcessPass};
pub use graph::{
	input_from_chain, ExtraInput, NodeId, NodeInput, PostProcessGraph,
	PostProcessNode,
};
pub use quad::{fullscreen_render_type, FullscreenQuad};
