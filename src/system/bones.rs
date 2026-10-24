// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 骨骼矩阵的存储缓冲.
//!
//! [`BoneBuffer`] 是一个 `storage buffer`, 存放所有实例的骨骼矩阵, 用于
//! GPU 蒙皮 —— 顶点着色器先通过 `@builtin(instance_index)` 定位实例,
//! 再结合实例的 `bone_offset` 与顶点自带的 `bone_index` 定位到实际使用
//! 的骨骼矩阵.
//!
//! # 与 `InstanceBuffer` 的关系
//!
//! 两者分工配合:
//!
//! - [`crate::system::InstanceBuffer`] —— 存放每个实例的 `model` 矩阵
//!   以及 `bone_offset` / `bone_count`.
//! - [`BoneBuffer`] —— 存放所有实例的骨骼矩阵, 按实例连续排列.
//!
//! 某个实例的骨骼矩阵在缓冲中的区间为
//! `[bone_offset, bone_offset + bone_count)`.
//!
//! # 帧间复用
//!
//! [`BoneBuffer::clear`] 重置使用计数, GPU 缓冲区仍然保留. 跨帧复用
//! 时无需重新分配.
//!
//! # 容量
//!
//! 容量在创建时固定. 超出容量时 [`BoneBuffer::push_skeleton`]
//! 返回 [`crate::system::Error::BoneBufferFull`].

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use crate::system::error::Result;

/// 单个骨骼的变换矩阵, 与 [`Mat4`] 二进制兼容 —— 4 列, 每列 4 个 `f32`.
///
/// # 布局
///
/// 采用 `#[repr(C)]` 布局并实现 `Pod` —— 可按原始字节直接上传.
///
/// # 坐标约定
///
/// 骨骼矩阵刻画的是**模型空间内的变形**, 最终顶点位置为
/// `instance.model * bone_matrix * vertex_pos`.
///
/// 这意味着 `bone_matrix` 同时包含两类信息:
///
/// - 骨骼在模型空间的位置 (相对骨骼原点的变换).
/// - 动画带来的姿态 (旋转, 平移, 缩放).
///
/// 该约定参考了 Minecraft 风格的骨骼模型.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BoneMatrix {
	/// 列主序存放的 4x4 矩阵.
	pub matrix: [[f32; 4]; 4],
}

impl BoneMatrix {
	/// 构造单位骨骼矩阵.
	pub fn identity() -> Self {
		Self {
			matrix: Mat4::IDENTITY.to_cols_array_2d(),
		}
	}
}

impl From<Mat4> for BoneMatrix {
	fn from(m: Mat4) -> Self {
		Self {
			matrix: m.to_cols_array_2d(),
		}
	}
}

/// 骨骼矩阵的 `storage buffer`.
///
/// # 布局
///
/// 所有实例的骨骼矩阵**紧密**排列, 中间没有 `padding`. 每个实例覆盖的
/// 骨骼区间由 [`crate::system::InstanceData`] 的 `bone_offset` /
/// `bone_count` 两个字段决定.
///
/// # 对齐
///
/// `size_of::<BoneMatrix>()` 为 64 字节 (4x4 个 `f32`), 是 16 的
/// 倍数 —— 满足 `storage buffer` 的对齐要求.
pub struct BoneBuffer {
	buffer: wgpu::Buffer,
	bind_group: wgpu::BindGroup,
	bind_group_layout: wgpu::BindGroupLayout,
	capacity: u32,
	used: u32,
}

impl BoneBuffer {
	/// 新建一个骨骼缓冲.
	///
	/// # 参数
	///
	/// - `device` —— 用于创建缓冲与 `bind group`.
	/// - `capacity` —— 跨所有实例的最大骨骼数; 为 0 时提升到 1.
	pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
		let capacity = capacity.max(1);
		let stride = std::mem::size_of::<BoneMatrix>() as u64;

		// `storage buffer` 不使用 `dynamic offset` —— 只需 4 字节对齐.
		//
		// **保守按 256 对齐**: 256 是
		// `min_uniform_buffer_offset_alignment` 的常见值, 对
		// `storage buffer` 并非必需. 多分配几十到几百字节的代价可
		// 忽略, 换来「将来若改用 `dynamic offset` 时无需重新审计
		// 对齐要求」.
		//
		// 需要精确尺寸时改成 4 —— 但若将来引入 `dynamic offset`,
		// 记得先查 `Limits::min_storage_buffer_offset_alignment`.
		let raw_size = (capacity as u64) * stride;
		let size = raw_size.div_ceil(256) * 256;

		let buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("bone_buffer"),
			size,
			usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let bind_group_layout =
			device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
				label: Some("bone_bind_group_layout"),
				entries: &[wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::VERTEX,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Storage { read_only: true },
						has_dynamic_offset: false,
						min_binding_size: None,
					},
					count: None,
				}],
			});

		let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("bone_bind_group"),
			layout: &bind_group_layout,
			entries: &[wgpu::BindGroupEntry {
				binding: 0,
				resource: buffer.as_entire_binding(),
			}],
		});

		Self {
			buffer,
			bind_group,
			bind_group_layout,
			capacity,
			used: 0,
		}
	}

	// ================ 帧计数 ================

	/// 把使用计数清零.
	///
	/// 每帧开始时调用一次, 让 [`Self::push_skeleton`] 从索引 0 重新
	/// 开始分配.
	pub fn clear(&mut self) {
		self.used = 0;
	}

	/// 本帧到目前为止已使用的骨骼数.
	pub fn len(&self) -> u32 {
		self.used
	}

	/// 判断本帧是否尚未上传任何骨骼.
	pub fn is_empty(&self) -> bool {
		self.used == 0
	}

	/// 缓冲能容纳的最大骨骼数.
	pub fn capacity(&self) -> u32 {
		self.capacity
	}

	// ================ 绑定资源 ================

	/// 用于绑定的 `bind group`.
	pub fn bind_group(&self) -> &wgpu::BindGroup {
		&self.bind_group
	}

	/// 上述 `bind group` 的布局.
	pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
		&self.bind_group_layout
	}

	// ================ 骨骼上传 ================

	/// 追加一组骨骼矩阵.
	///
	/// # 返回值
	///
	/// 返回这组骨骼在缓冲中的**起始索引** (并非字节偏移), 它应作为
	/// [`crate::system::InstanceBuffer::push_skinned`] 的 `bone_offset` 传入.
	///
	/// # 错误
	///
	/// 剩余容量不足时返回 [`crate::system::Error::BoneBufferFull`].
	///
	/// # 空数组
	///
	/// `bones` 为空时直接返回当前的 `used`, 不占用任何槽位. 调用方可以
	/// 把它当作「无骨骼」的合法输入.
	pub fn push_skeleton(
		&mut self,
		queue: &wgpu::Queue,
		bones: &[Mat4],
	) -> Result<u32> {
		let count = bones.len() as u32;
		if count == 0 {
			return Ok(self.used);
		}

		// 这里用减法而不是 `self.used + count > self.capacity`: 后者在
		// `capacity` 接近 `u32::MAX` 时可能回绕, 使检查通过但计数错乱.
		// 本类型的内部不变量是 `self.used <= self.capacity` (`used` 只在
		// 检查通过后才递增), 所以减法不会下溢.
		//
		// **不要**改成 `self.used + count > self.capacity`: 在极端容量下
		// 它会静默放行, 后续写入便会落到错误的偏移.
		if count > self.capacity - self.used {
			return Err(crate::system::Error::BoneBufferFull);
		}

		let offset = (self.used as u64) * std::mem::size_of::<BoneMatrix>() as u64;

		// 先把整个切片转换一次再上传 —— `bytemuck` 需要 `&[BoneMatrix]`,
		// 而输入是 `&[Mat4]`, 因此要付一次拷贝; 骨骼数量通常很少
		// (几十根), 这点开销可以忽略.
		//
		// **不要**改成对每个 `Mat4` 单独调用 `write_buffer`: 那样会产生
		// `count` 次提交, 每次都有固定开销; 用一次 `write_buffer` 处理
		// 整个切片效率更高.
		let matrices: Vec<BoneMatrix> = bones.iter().copied().map(Into::into).collect();
		queue.write_buffer(&self.buffer, offset, bytemuck::cast_slice(&matrices));

		let start = self.used;
		self.used += count;
		Ok(start)
	}
}
