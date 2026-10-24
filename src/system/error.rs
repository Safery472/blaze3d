// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 渲染系统的各类错误.
//!
//! 汇总 [`crate::system`] 模块内所有可能失败的公开 API.

use thiserror::Error;

/// 渲染系统各类操作可能返回的错误.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum Error {
	// ================ 网格操作 ================

	/// 待上传的网格不含任何顶点.
	#[error("mesh has no vertices")]
	EmptyMesh,

	/// 更新网格时, 新数据的布局与原始布局不一致.
	#[error("mesh layout mismatch: {0}")]
	MeshLayoutMismatch(&'static str),

	// ================ 对象 `uniform` 缓冲 ================

	/// 对象缓冲的槽位已经用尽.
	#[error("object buffer is full")]
	ObjectBufferFull,

	// ================ 实例数据缓冲 ================

	/// 实例缓冲的容量已经用尽.
	#[error("instance buffer is full")]
	InstanceBufferFull,

	// ================ 骨骼矩阵缓冲 ================

	/// 骨骼缓冲的剩余容量不足.
	#[error("bone buffer is full")]
	BoneBufferFull,

	// ================ GPU 时间线 ================

	/// 设备不具备 `timestamp` 查询能力.
	///
	/// 由 [`crate::system::RenderSystem::enable_timeline`] 返回, 前提是
	/// 设备缺少 [`wgpu::Features::TIMESTAMP_QUERY`].
	///
	/// # 应对
	///
	/// - 申请设备时显式启用该特性.
	/// - 或者接受没有 GPU 计时 —— 调用方应捕获此错误并降级.
	#[error("device does not support TIMESTAMP_QUERY feature")]
	TimestampQueryNotSupported,

	/// 时间线尚未启用.
	///
	/// 由 [`crate::system::RenderSystem::read_timeline`] 返回. 读取之前必须
	/// 先调用 `enable_timeline` 启用时间线.
	#[error("timeline is not enabled")]
	NoTimeline,

	/// 时间线数据读回失败.
	///
	/// 由 [`crate::system::RenderSystem::read_timeline`] 返回. 常见原因是
	/// 设备丢失或后端报错.
	#[error("timeline readback failed")]
	TimelineMapFailed,

	// ================ 骨架与动画轨道 ================

	/// 骨架不含任何骨骼.
	///
	/// 由 [`crate::system::Skeleton::new`] 返回.
	#[error("skeleton has no bones")]
	EmptySkeleton,

	/// 骨骼的父索引不合法.
	///
	/// 由 [`crate::system::Skeleton::new`] 返回. 父骨骼索引必须**小于**
	/// 子骨骼索引 —— 这样 `compute_matrices` 只需单趟前向遍历即可完成.
	#[error("bone {bone} has invalid parent {parent} (parent must precede the bone)")]
	InvalidBoneParent {
		/// 子骨骼的索引.
		bone: usize,
		/// 请求指定的父骨骼索引.
		parent: usize,
	},

	/// 骨骼的绑定矩阵无法求逆.
	///
	/// 由 [`crate::system::Skeleton::new`] 返回. 绑定姿态退化 (行列式
	/// 接近 0) 时求逆会失败, 因而无法计算骨骼矩阵.
	#[error("bone {bone} has a singular bind matrix")]
	SingularBoneBind {
		/// 该骨骼的索引.
		bone: usize,
	},

	/// 姿态包含的骨骼数与骨架不一致.
	///
	/// 由 [`crate::system::Skeleton::compute_matrices`] 返回.
	#[error("pose has {actual} bones but skeleton has {expected}")]
	PoseLengthMismatch {
		/// 骨架包含的骨骼数.
		expected: usize,
		/// 姿态包含的骨骼数.
		actual: usize,
	},

	/// 动画轨道的关键帧时间数与变换数不一致.
	///
	/// 由 [`crate::system::BoneTrack::new`] 返回.
	#[error("track has {times} keyframe times but {transforms} transforms")]
	TrackLengthMismatch {
		/// 关键帧时间的数量.
		times: usize,
		/// 变换的数量.
		transforms: usize,
	},

	/// 动画轨道的关键帧时间不是严格升序.
	///
	/// 由 [`crate::system::BoneTrack::new`] 返回. 采样依赖升序排列做
	/// 二分查找.
	#[error("bone {bone} track keyframe times are not strictly increasing")]
	TrackNotMonotonic {
		/// 该骨骼的索引.
		bone: usize,
	},
}

/// `system` 模块通用的 `Result` 别名.
pub type Result<T, E = Error> = std::result::Result<T, E>;
