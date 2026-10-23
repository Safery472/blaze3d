//! blaze3d —— 基于 wgpu 的 3D 渲染库.
//!
//! 本 crate 提供 Minecraft 风格渲染所需的基础设施: 相机, 顶点数据构建,
//! 纹理与绑定资源, 管线状态与缓存, 渲染目标, 后处理, 以及把上述资源组织
//! 成每帧渲染流程的 [`system::RenderSystem`].
//!
//! # 模块
//!
//! - [`camera`] —— 相机抽象 ([`camera::Camera`] / [`camera::PerspectiveCamera`] /
//!   [`camera::OrthographicCamera`]) 与视锥剔除.
//! - [`vertex`] —— 顶点格式, 顶点构建器, 索引与实例数据.
//! - [`texture`] —— 纹理, 采样器, bind group 构建与 mipmap 生成.
//! - [`state`] —— 管线状态描述, 颜色 / 深度格式与管线缓存.
//! - [`shader`] —— WGSL 着色器模块与内置着色器.
//! - [`pipeline`] —— 渲染目标, 深度 / MSAA 缓冲, 窗口表面, 阴影贴图, 读回.
//! - [`post`] —— 后处理链与后处理图.
//! - [`system`] —— 长期 GPU 资源与每帧渲染流程.
//!
//! # 快速开始
//!
//! [`prelude`] 重导出了常用类型, 推荐用一条 `use blaze3d::prelude::*;`
//! 引入. 需要更细粒度的控制时, 直接 `use` 对应模块.
//!
//! # 与 Java 版 `Blaze3D` 的关系
//!
//! 模块划分参考 Java 版 `com.mojang.blaze3d`, 但按 wgpu 的能力重新设计:
//! GPU 资源由调用方显式持有, 管线按描述惰性缓存, 每帧流程由 [`system`]
//! 组织.
//!
//! # 安全
//!
//! crate 级别禁止 `unsafe` (见 `#![deny(unsafe_code)]`).

#![deny(unsafe_code)]

pub mod camera;
pub mod pipeline;
pub mod post;
pub mod shader;
pub mod state;
pub mod system;
pub mod texture;
pub mod vertex;

pub mod prelude {
	//! 常用类型重导出.
	//!
	//! 覆盖渲染应用最常见的入口类型, 便于用一条 `use blaze3d::prelude::*;`
	//! 引入. 需要更细粒度的控制时, 直接 `use` 对应模块.

	pub use crate::camera::{
		Aabb, Camera, Frustum, OrthographicCamera, PerspectiveCamera,
	};

	pub use crate::shader::Shader;
	pub use crate::shader::{BuiltinShader, BUILTIN_ID_BASE};

	pub use crate::state::{
		BlendMode, ColorFormat, CompareOp, ComputePipelineDesc, CullMode,
		DepthFormat, DepthMode, LayoutHandle, PolygonMode, RenderType,
		RenderTypeBuilder, WriteMask,
	};
	pub use crate::state::{PipelineCache, PipelineDesc};

	pub use crate::system::{
		AnimationClip, Bone, BoneBuffer, BoneMatrix, BoneTrack, BoneTransform,
		ComputePassHelper, DrawItem, Frame, GpuMesh, InstanceBuffer,
		InstanceData, ObjectBuffer, ObjectUniforms, RenderList, RenderSystem,
		RenderSystemDescriptor, SharedGpuMesh, SharedIndexBuffers, Skeleton,
		SkeletonPose, Timeline,
	};
	pub use crate::system::PassHelper;

	pub use crate::pipeline::{CubeFace, CubeShadowMap, RenderTarget, ShadowMap, Target};

	pub use crate::post::{
		ExtraInput, FullscreenQuad, NodeId, NodeInput, PostProcessChain,
		PostProcessGraph, PostProcessInput, PostProcessPass,
	};

	pub use crate::texture::{
		BindGroupBuilder, BindGroupLayoutBuilder, Sampler, Texture, TextureDesc,
		TextureKind,
	};

	pub use crate::vertex::{
		BufferBuilder, Building, DrawMode, RenderedBuffer, UploadedBuffer,
		VertexConsumer, VertexFormat, VertexFormatElement,
	};
	pub use crate::vertex::formats;
	pub use crate::vertex::PoseStack;
}
