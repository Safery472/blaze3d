//! 渲染系统的长期资源与每帧流程.
//!
//! 本模块持有渲染流程所需的长期资源, 并协调每一帧的渲染顺序.
//!
//! # 模块结构
//!
//! [`RenderSystem`] 是全部长期资源的持有者. 渲染入口共有三种:
//!
//! - [`RenderSystem::render_frame`] —— 单 pass 的便捷封装.
//! - [`RenderSystem::begin_frame`] + [`Frame::pass`] + [`Frame::submit`] ——
//!   多 pass 渲染.
//! - [`Frame::compute_pass`] —— 计算 pass.
//!
//! 常用类型如下:
//!
//! - [`RenderSystem`] —— 资源持有者.
//! - [`Frame`] —— 一帧的渲染上下文.
//! - [`PassHelper`] —— 单次 render pass 的绘制接口.
//! - [`ComputePassHelper`] —— 单次 compute pass 的 dispatch 接口.
//! - [`RenderList`] —— 绘制项的收集与排序.
//! - [`Globals`] —— 帧级 `uniform`.
//! - [`ObjectBuffer`] —— 每对象 `uniform` 的环形缓冲.
//! - [`InstanceBuffer`] —— 每实例数据的存储缓冲.
//! - [`BoneBuffer`] —— 骨骼矩阵的存储缓冲.
//! - [`Skeleton`] —— 骨骼层级与动画采样 (骨架动画的完整设施).
//! - [`SharedIndexBuffers`] —— 顺序索引缓冲.
//! - [`GpuMesh`] —— 长驻 GPU 的网格缓冲.
//! - [`Timeline`] —— GPU 时间线查询.
//!
//! # 时间线
//!
//! 启用 [`RenderSystem::enable_timeline`] 之后, 每个 pass 都会自动记录
//! GPU 耗时. 细节见 [`Timeline`].

// ================ 子模块声明 ================

pub mod bones;
pub mod error;
pub mod indices;
pub mod instances;
pub mod list;
pub mod mesh;
pub mod objects;
pub mod renderer;
pub mod skeleton;
pub mod timeline;

// ================ 公开重导出 ================

// -- 错误类型 --

pub use error::{Error, Result};

// -- 资源类型 --

pub use bones::{BoneBuffer, BoneMatrix};
pub use indices::SharedIndexBuffers;
pub use instances::{InstanceBuffer, InstanceData};
pub use list::{DrawItem, RenderList};
pub use mesh::{GpuMesh, SharedGpuMesh};
pub use objects::{ObjectBuffer, ObjectUniforms};
pub use skeleton::{
	AnimationClip, Bone, BoneTrack, BoneTransform, Skeleton, SkeletonPose,
};
pub use timeline::Timeline;

// -- 渲染器 --

pub use renderer::{
	ComputePassHelper, Frame, Globals, PassHelper, RenderSystem, RenderSystemDescriptor,
};
