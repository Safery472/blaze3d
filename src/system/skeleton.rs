// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 骨架与关键帧动画的采样.
//!
//! 本模块在 [`crate::system::bones::BoneBuffer`] 之上补齐完整的骨架动画
//! 设施:
//!
//! - [`Skeleton`] —— 骨骼层级, 即父子关系与绑定姿态.
//! - [`SkeletonPose`] —— 一组骨骼的局部变换, 逐骨骼以 TRS 表示.
//! - [`AnimationClip`] 与 [`BoneTrack`] —— 关键帧动画的数据结构.
//! - [`Skeleton::compute_matrices`] —— 从局部姿态导出最终骨骼矩阵
//!   (`bind_inverse * global_pose`), 结果可直接上传到
//!   [`crate::system::bones::BoneBuffer`].
//!
//! # 与 `BoneBuffer` 的关系
//!
//! [`crate::system::bones::BoneBuffer`] 是**纯 GPU 资源** —— 它只保存
//! 矩阵, 既不知道骨骼的父子关系, 也不做坐标系换算. 这些矩阵由本模块
//! 负责**算出**. 典型流程:
//!
//! ```ignore
//! // 启动阶段: 定义骨架.
//! let skeleton = Skeleton::new(
//!     vec![
//!         Bone { parent: None,     bind: Mat4::IDENTITY },          // 根
//!         Bone { parent: Some(0),  bind: Mat4::from_translation(..) }, // 上臂
//!         Bone { parent: Some(1),  bind: Mat4::from_translation(..) }, // 前臂
//!     ],
//! );
//! let mut pose = skeleton.rest_pose();
//! let clip = /* 从文件加载的关键帧 */;
//!
//! // 每帧: 采样动画 → 算矩阵 → 上传.
//! clip.sample(time, &mut pose);
//! let matrices = skeleton.compute_matrices(&pose);
//! let offset = bone_buffer.push_skeleton(&queue, &matrices)?;
//! instances.push_skinned(&queue, model, offset, matrices.len() as u32)?;
//! ```
//!
//! # 坐标系约定
//!
//! 骨骼矩阵是**模型空间内的变形**: 顶点着色器里按
//! `model * bone_matrix * vertex_pos` 组合. 详见
//! [`crate::system::bones::BoneMatrix`] 的文档. `bind` 存的是绑定姿态的
//! 逆 —— 也就是骨骼空间到模型空间的变换的逆. `compute_matrices` 内部
//! 用 `bind_inverse * global_pose` 的形式, 因此 [`Bone::bind`] 填的是
//! **绑定姿态的世界矩阵**, 调用方不必预先求逆.
//!
//! # 限制
//!
//! - **顶点格式** —— 蒙皮的顶点布局要包含 `BoneIndex` 与 `BoneWeight`
//!   元素, 见 [`crate::vertex::formats::position_color_skinned`].
//! - **每顶点骨骼数** —— 固定为 4; 权重通常归一化到总和为 1.
//! - **插值** —— 位置与缩放按线性插值, 旋转按球面插值.

use glam::{Mat4, Quat, Vec3};

use super::error::{Error, Result};

// ================ 骨骼定义 ================

/// 骨架中的单根骨骼.
///
/// # `bind`
///
/// 绑定姿态下该骨骼的**世界矩阵** (模型空间).
/// [`Skeleton::compute_matrices`] 内部会对它求逆 —— 调用方不必事先
/// 算好逆矩阵.
#[derive(Debug, Clone, Copy)]
pub struct Bone {
	/// 父骨骼的索引; `None` 表示它就是根骨骼.
	pub parent: Option<usize>,
	/// 绑定姿态下该骨骼的世界矩阵.
	pub bind: Mat4,
}

// ================ 骨架结构 ================

/// 骨架 —— 骨骼构成的层级结构.
///
/// # 不变量
///
/// - 骨骼的 `parent` 若存在, 必须是**更小**的索引. 只有这样才能让
///   [`Self::compute_matrices`] 一趟前向遍历算完 —— 索引序本身就是
///   拓扑序.
/// - 骨架非空时, 索引 0 通常就是根骨骼.
#[derive(Debug, Clone)]
pub struct Skeleton {
	bones: Vec<Bone>,
	/// 各骨骼绑定姿态矩阵的逆, 在构造时一次算好.
	bind_inverses: Vec<Mat4>,
}

impl Skeleton {
	/// 用给定的骨骼列表创建骨架.
	///
	/// # 错误
	///
	/// - 骨骼列表为空 —— [`Error::EmptySkeleton`].
	/// - 某骨骼的 `parent >= 自身索引` —— [`Error::InvalidBoneParent`].
	///   父骨骼必须排在子骨骼之前, 见类型文档中的不变量.
	/// - 某骨骼的 `bind` 不可逆 —— [`Error::SingularBoneBind`].
	pub fn new(bones: Vec<Bone>) -> Result<Self> {
		if bones.is_empty() {
			return Err(Error::EmptySkeleton);
		}

		let mut bind_inverses = Vec::with_capacity(bones.len());
		for (i, bone) in bones.iter().enumerate() {
			if let Some(parent) = bone.parent
				&& parent >= i
			{
				return Err(Error::InvalidBoneParent {
					bone: i,
					parent,
				});
			}

			let det = bone.bind.determinant();
			if det.abs() < f32::EPSILON {
				return Err(Error::SingularBoneBind { bone: i });
			}
			bind_inverses.push(bone.bind.inverse());
		}

		Ok(Self {
			bones,
			bind_inverses,
		})
	}

	/// 返回骨骼总数.
	pub fn len(&self) -> usize {
		self.bones.len()
	}

	/// 返回骨架是否为空. 恒为 `false` —— 空骨架在 [`Self::new`] 就被
	/// 拒绝.
	pub fn is_empty(&self) -> bool {
		self.bones.is_empty()
	}

	/// 按索引取出一根骨骼.
	pub fn bone(&self, index: usize) -> Option<&Bone> {
		self.bones.get(index)
	}

	/// 由绑定姿态推出各骨骼的局部变换.
	///
	/// # 行为
	///
	/// 从 `bind` 世界矩阵反推: 根骨骼的局部变换恰好就是它的 `bind`;
	/// 子骨骼的局部变换为 `parent_bind^-1 * bind`.
	pub fn rest_pose(&self) -> SkeletonPose {
		let mut locals = Vec::with_capacity(self.bones.len());

		for bone in &self.bones {
			let local = match bone.parent {
				None => bone.bind,
				Some(parent) => self.bind_inverses[parent] * bone.bind,
			};
			let (scale, rotation, translation) = local.to_scale_rotation_translation();
			locals.push(BoneTransform {
				translation,
				rotation,
				scale,
			});
		}

		SkeletonPose { locals }
	}

	/// 由局部姿态算出最终的骨骼矩阵.
	///
	/// # 算法
	///
	/// 按索引序做单趟前向遍历 (索引序即拓扑序):
	///
	/// ```text
	/// global[i] = global[parent] * local[i]      (根: global[i] = local[i])
	/// matrix[i] = bind_inverse[i] * global[i]
	/// ```
	///
	/// # 前置条件
	///
	/// 要求 `pose.locals.len() == self.len()`; 长度不符时返回
	/// [`Error::PoseLengthMismatch`].
	///
	/// # 返回
	///
	/// 每根骨骼对应一个 [`Mat4`], 顺序与骨架一致, 可以直接交给
	/// [`BoneBuffer::push_skeleton`].
	///
	/// [`BoneBuffer::push_skeleton`]: crate::system::BoneBuffer::push_skeleton
	pub fn compute_matrices(&self, pose: &SkeletonPose) -> Result<Vec<Mat4>> {
		if pose.locals.len() != self.bones.len() {
			return Err(Error::PoseLengthMismatch {
				expected: self.bones.len(),
				actual: pose.locals.len(),
			});
		}

		let mut globals = vec![Mat4::IDENTITY; self.bones.len()];
		let mut out = vec![Mat4::IDENTITY; self.bones.len()];

		for i in 0..self.bones.len() {
			let local = pose.locals[i].to_matrix();
			let global = match self.bones[i].parent {
				None => local,
				Some(parent) => globals[parent] * local,
			};
			globals[i] = global;
			out[i] = self.bind_inverses[i] * global;
		}

		Ok(out)
	}

	/// 返回某根骨骼的全局变换, 由局部姿态逐级累乘得到.
	///
	/// 骨骼附加物 (例如挂到手上的武器) 需要这类变换.
	///
	/// # 返回
	///
	/// `index` 超出骨架或姿态的范围时返回 `None`.
	pub fn bone_matrix(&self, pose: &SkeletonPose, index: usize) -> Option<Mat4> {
		if index >= self.bones.len() || index >= pose.locals.len() {
			return None;
		}

		let mut chain = Vec::new();
		let mut current = Some(index);
		while let Some(i) = current {
			chain.push(i);
			current = self.bones[i].parent;
		}

		let mut global = Mat4::IDENTITY;
		for &i in chain.iter().rev() {
			global *= pose.locals[i].to_matrix();
		}
		Some(global)
	}
}

// ================ 姿态类型 ================

/// 单根骨骼的局部变换, 以 TRS 表示.
///
/// 之所以不直接存矩阵, 是因为动画插值要对平移, 旋转与缩放分别
/// 插值; 旋转用四元数球面插值 (slerp) 可以避开万向锁.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoneTransform {
	/// 局部平移量.
	pub translation: Vec3,
	/// 局部旋转, 以四元数表示.
	pub rotation: Quat,
	/// 局部缩放系数.
	pub scale: Vec3,
}

impl BoneTransform {
	/// 恒等变换 —— 不平移, 不旋转, 缩放为 1.
	pub const fn identity() -> Self {
		Self {
			translation: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::ONE,
		}
	}

	/// 把 TRS 合成为变换矩阵.
	pub fn to_matrix(self) -> Mat4 {
		Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
	}

	/// 在两个变换之间插值.
	///
	/// # 参数
	///
	/// - `other` —— 目标变换.
	/// - `t` —— 插值因子, 取值 `[0, 1]`.
	///
	/// # 插值方式
	///
	/// - 平移与缩放 —— 线性插值.
	/// - 旋转 —— 球面插值 (slerp).
	pub fn lerp(self, other: Self, t: f32) -> Self {
		Self {
			translation: self.translation.lerp(other.translation, t),
			rotation: self.rotation.slerp(other.rotation, t),
			scale: self.scale.lerp(other.scale, t),
		}
	}
}

/// 一组骨骼的局部姿态, 顺序与骨架对齐.
#[derive(Debug, Clone)]
pub struct SkeletonPose {
	/// 各骨骼的局部变换, 顺序与骨架一致.
	pub locals: Vec<BoneTransform>,
}

impl SkeletonPose {
	/// 返回姿态中包含的骨骼数量.
	pub fn len(&self) -> usize {
		self.locals.len()
	}

	/// 返回姿态是否为空.
	pub fn is_empty(&self) -> bool {
		self.locals.is_empty()
	}

	/// 就地覆盖某根骨骼的局部变换; 索引越界时静默忽略.
	pub fn set(&mut self, index: usize, local: BoneTransform) {
		if let Some(slot) = self.locals.get_mut(index) {
			*slot = local;
		}
	}

	/// 读取某根骨骼的局部变换.
	pub fn get(&self, index: usize) -> Option<BoneTransform> {
		self.locals.get(index).copied()
	}
}

// ================ 关键帧动画 ================

/// 单根骨骼的关键帧轨道.
///
/// 关键帧的时间戳按**升序**存放; [`Self::sample`] 用二分查找定位
/// 区间, 再做插值.
#[derive(Debug, Clone)]
pub struct BoneTrack {
	/// 该轨道作用的骨骼索引.
	pub bone: usize,
	/// 各关键帧的时间戳 (秒), 按升序排列.
	pub times: Vec<f32>,
	/// 各关键帧的局部变换, 长度与 `times` 相同.
	pub transforms: Vec<BoneTransform>,
}

impl BoneTrack {
	/// 创建一条关键帧轨道.
	///
	/// # 错误
	///
	/// - `times` 与 `transforms` 长度不一致 ——
	///   [`Error::TrackLengthMismatch`].
	/// - `times` 不是严格升序 —— [`Error::TrackNotMonotonic`].
	pub fn new(bone: usize, times: Vec<f32>, transforms: Vec<BoneTransform>) -> Result<Self> {
		if times.len() != transforms.len() {
			return Err(Error::TrackLengthMismatch {
				times: times.len(),
				transforms: transforms.len(),
			});
		}
		if times.windows(2).any(|w| w[1] <= w[0]) {
			return Err(Error::TrackNotMonotonic { bone });
		}
		Ok(Self {
			bone,
			times,
			transforms,
		})
	}

	/// 在给定时刻采样出一个局部变换.
	///
	/// # 行为
	///
	/// - 早于第一帧的时间 —— 返回第一帧.
	/// - 晚于最后一帧的时间 —— 返回最后一帧.
	/// - 落在两帧之间 —— 按线性与球面插值求出中间值.
	/// - 空轨道 —— 返回 `None`.
	///
	/// # 返回
	///
	/// 轨道没有关键帧 (`times` 为空) 时返回 `None`.
	pub fn sample(&self, time: f32) -> Option<BoneTransform> {
		if self.times.is_empty() {
			return None;
		}

		// 落在时间范围之外 —— 直接钳制到端点帧.
		if time <= self.times[0] {
			return Some(self.transforms[0]);
		}
		let last = self.times.len() - 1;
		if time >= self.times[last] {
			return Some(self.transforms[last]);
		}

		// 二分查找第一个大于 time 的索引.
		let next = self.times.partition_point(|&t| t <= time);
		let prev = next - 1;

		let t0 = self.times[prev];
		let t1 = self.times[next];
		let span = t1 - t0;
		let factor = if span > 0.0 { (time - t0) / span } else { 0.0 };

		Some(self.transforms[prev].lerp(self.transforms[next], factor))
	}
}

/// 一段由关键帧组成的动画.
///
/// 每条 [`BoneTrack`] 对应一根骨骼; 一段动画只改它带有轨道的那些
/// 骨骼, 其余骨骼保持原样.
#[derive(Debug, Clone)]
pub struct AnimationClip {
	/// 动画的名称.
	pub name: String,
	/// 动画时长 (秒), 循环播放时用它取模.
	pub duration: f32,
	/// 各条骨骼轨道.
	pub tracks: Vec<BoneTrack>,
}

impl AnimationClip {
	/// 创建一段动画.
	///
	/// # 时长推导
	///
	/// `duration` 为 0 或负数时, 改由各轨道中最晚的关键帧推导.
	pub fn new(name: impl Into<String>, duration: f32, tracks: Vec<BoneTrack>) -> Self {
		let duration = if duration > 0.0 {
			duration
		} else {
			tracks
				.iter()
				.filter_map(|t| t.times.last().copied())
				.fold(0.0_f32, f32::max)
		};

		Self {
			name: name.into(),
			duration,
			tracks,
		}
	}

	/// 返回动画时长 (秒).
	pub fn duration(&self) -> f32 {
		self.duration
	}

	/// 在给定时刻采样整段动画, 结果写回姿态.
	///
	/// # 行为
	///
	/// 逐条轨道采样并写入 `pose.locals[track.bone]`; 没有任何轨道覆盖
	/// 的骨骼保留 `pose` 中的当前值.
	///
	/// # 时间处理
	///
	/// - `looping == true` —— 时间对 `duration` 取模, 即循环播放.
	/// - `looping == false` —— 时间钳制到 `[0, duration]`.
	///
	/// # 返回值
	///
	/// 实际写入的轨道条数.
	pub fn sample(&self, time: f32, looping: bool, pose: &mut SkeletonPose) -> usize {
		let t = if self.duration > 0.0 {
			if looping {
				time.rem_euclid(self.duration)
			} else {
				time.clamp(0.0, self.duration)
			}
		} else {
			time
		};

		let mut applied = 0;
		for track in &self.tracks {
			if let Some(transform) = track.sample(t)
				&& let Some(slot) = pose.locals.get_mut(track.bone)
			{
				*slot = transform;
				applied += 1;
			}
		}
		applied
	}

	/// 把整段动画的关键帧时间归一到 `[0, 1]`.
	///
	/// 供需要做动画重定向或混合的调用方使用.
	pub fn normalized_tracks(&self) -> Vec<(usize, Vec<(f32, BoneTransform)>)> {
		if self.duration <= 0.0 {
			return Vec::new();
		}
		self.tracks
			.iter()
			.map(|t| {
				let frames = t
					.times
					.iter()
					.zip(t.transforms.iter())
					.map(|(&time, &tr)| (time / self.duration, tr))
					.collect();
				(t.bone, frames)
			})
			.collect()
	}
}
