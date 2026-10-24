// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 渲染目标与窗口表面的错误.
//!
//! 本模块覆盖 [`crate::pipeline`] 内所有可能失败的公开 API. 与 `vertex` /
//! `system` / `texture` 保持一致: 每个子系统只暴露一个错误枚举, 调用方
//! 按变体集中匹配.

use thiserror::Error;

/// 渲染目标与窗口表面操作可能返回的错误.
///
/// 枚举标注了 `#[non_exhaustive]` —— 后续版本可能新增变体, 下游的
/// `match` 必须保留 `_` 分支.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum Error {
	// ================ 目标创建 ================

	/// 请求的纹理尺寸超过设备上限.
	///
	/// 上限来自设备的 `max_texture_dimension_2d`.
	#[error("texture size {requested_width}x{requested_height} exceeds device limit {limit}")]
	SizeExceedsLimit {
		/// 请求的宽度.
		requested_width: u32,
		/// 请求的高度.
		requested_height: u32,
		/// 设备的 `max_texture_dimension_2d`.
		limit: u32,
	},

	/// 请求的尺寸为 0.
	///
	/// 颜色纹理, 深度纹理与 MSAA 缓冲都要求宽高非零.
	#[error("texture size must be non-zero (got {width}x{height})")]
	ZeroSize {
		/// 请求的宽度.
		width: u32,
		/// 请求的高度.
		height: u32,
	},

	/// 采样数非法.
	///
	/// MSAA 缓冲的采样数必须 >= 2. 需要单采样时应当改走非 MSAA 路径,
	/// 而不是传 1.
	#[error("invalid sample count {count} (MSAA requires >= 2)")]
	InvalidSampleCount {
		/// 请求的采样数.
		count: u32,
	},

	/// 阴影光源的范围非法.
	///
	/// 由 [`crate::pipeline::CubeShadowMap::new`] 返回. 范围必须是有限
	/// 正数 —— 它同时作为透视投影的远平面.
	///
	/// # 为什么不含 `f32` 字段
	///
	/// `f32` 不满足 [`Eq`], 而本枚举整体派生了 `Eq` (便于测试中直接
	/// `assert_eq!` 比较错误). 非法范围的具体值对定位问题帮助有限 ——
	/// 原因只有「非有限」或「非正」两种.
	#[error("invalid shadow light range (must be finite and > 0)")]
	InvalidShadowRange,

	// ================ 窗口表面 ================

	/// 获取表面纹理超时.
	///
	/// 通常意味着窗口表面暂时不可用; 下一帧重试即可.
	#[error("surface texture acquisition timed out")]
	SurfaceTimeout,

	/// 窗口被遮挡 (最小化或完全被其他窗口覆盖).
	#[error("surface is occluded (e.g. window minimized)")]
	SurfaceOccluded,

	/// 表面配置已过期.
	///
	/// 典型触发场景是窗口尺寸变化. 需要重新配置表面后再获取纹理.
	#[error("surface configuration is outdated (e.g. window resized)")]
	SurfaceOutdated,

	/// 表面丢失.
	///
	/// 通常意味着设备被移除或后端重启; 需要重建表面.
	#[error("surface was lost (e.g. device removed)")]
	SurfaceLost,

	/// 表面获取失败 (验证错误).
	///
	/// 具体原因见 wgpu 的日志输出.
	#[error("surface validation error (see wgpu logs for details)")]
	SurfaceValidation,

	/// 表面没有任何可用格式.
	#[error("surface has no supported format")]
	NoSurfaceFormat,

	/// 表面格式无法映射到 [`crate::state::ColorFormat`].
	#[error("surface format {format:?} is not supported by ColorFormat")]
	UnsupportedSurfaceFormat {
		/// 触发条件的 wgpu 格式.
		format: wgpu::TextureFormat,
	},

	// ================ 读回 ================

	/// 纹理格式不支持读回.
	///
	/// 由 [`crate::pipeline::read_texture`] 返回. 只支持未压缩的
	/// 2D 格式 —— 压缩格式 (BC / ETC / ASTC) 的块尺寸不是 1x1,
	/// 与「每像素字节数」的语义不符.
	#[error("texture format {format:?} is not supported for readback")]
	ReadbackUnsupportedFormat {
		/// 触发条件的格式.
		format: wgpu::TextureFormat,
	},

	/// 读回的缓冲映射失败.
	///
	/// 通常意味着设备丢失或后端错误. 具体原因见 wgpu 的日志输出.
	#[error("failed to map readback buffer")]
	ReadbackMapFailed,

	/// 等待 GPU 完成拷贝时失败.
	///
	/// 由 [`wgpu::Device::poll`] 返回错误时触发.
	#[error("device poll failed during readback")]
	ReadbackPollFailed,

	// ================ 后处理图 ================

	/// 后处理图中存在环.
	///
	/// 由 [`crate::post::PostProcessGraph::draw_into_frame`] 返回.
	/// 图必须是有向无环图 —— 环意味着某个节点间接依赖自己的输出.
	///
	/// # 排查
	///
	/// 检查每条 [`crate::post::NodeInput::Node`] 边 —— 存在一条从节点
	/// A 出发又回到 A 的路径即为环. 环检测在绘制时执行, 不在
	/// `add_node` 时执行 —— 见 [`crate::post::PostProcessGraph`] 的
	/// 「环检测」一节.
	#[error("post-process graph contains a cycle")]
	CycleDetected,

	/// 后处理图引用了不存在的节点.
	///
	/// 由 [`crate::post::PostProcessGraph::draw_into_frame`] 返回.
	/// [`crate::post::NodeInput::Node`] 的 id 超出了已添加节点的范围.
	#[error("post-process graph references non-existent node {id}")]
	InvalidNodeRef {
		/// 无效的节点 id.
		id: u32,
	},
}

/// `pipeline` 模块的 `Result` 别名.
pub type Result<T, E = Error> = std::result::Result<T, E>;
