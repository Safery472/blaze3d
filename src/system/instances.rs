// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 实例数据的存储缓冲.
//!
//! [`InstanceBuffer`] 是存放逐实例数据的 `storage buffer`, 服务于 GPU
//! 实例化绘制 —— 同一网格一次绘制多份, 每份使用不同的 `model` 矩阵.
//!
//! # 与 `ObjectBuffer` 的区别
//!
//! - [`crate::system::ObjectBuffer`] —— 每**绘制**一份 `uniform`, 通过
//!   `dynamic offset` 选择, 每次绘制都要调用 `set_bind_group`.
//! - [`InstanceBuffer`] —— 每**实例**一份数据, 通过
//!   `@builtin(instance_index)` 索引, 一次 `draw_instanced` 即可绘制
//!   全部实例.
//!
//! 前者适合「不同管线 / 不同纹理」的异构绘制; 后者适合
//! 「同管线同纹理」的批量绘制.
//!
//! # 蒙皮
//!
//! [`InstanceData`] 带有 `bone_offset` / `bone_count` 字段. 蒙皮渲染
//! 时, 顶点着色器靠这两个字段从 [`crate::system::BoneBuffer`] 中定位
//! 骨骼矩阵. 非蒙皮实例把两个字段都填 0 —— 着色器以
//! `bone_count == 0` 判断该实例是否需要蒙皮.
//!
//! # 使用
//!
//! ```ignore
//! // 启动.
//! let mut instances = InstanceBuffer::new(&device, 1024);
//!
//! // 每帧.
//! instances.clear();
//! for transform in transforms {
//!     instances.push(&queue, transform)?;
//! }
//!
//! system.render_frame(&target, |pass| {
//!     pass.set_render_type(&rt, &shader, &layout);
//!     pass.draw_mesh_instanced(&mesh, &instances);
//! });
//! ```
//!
//! # 着色器约定
//!
//! 实例化绘制的着色器必须满足:
//!
//! - Group 0 —— `Globals`.
//! - Group 1 —— `instances: array<Instance>` (storage, read-only).
//! - 顶点入口通过 `@builtin(instance_index)` 索引实例数组.
//!
//! 蒙皮渲染还需要额外的 `bind group`:
//!
//! - Group 2 —— `bones: array<mat4x4<f32>>` (storage, read-only).

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use crate::system::error::Result;

/// 一个实例携带的数据.
///
/// # 布局
///
/// 采用 `#[repr(C)]` 布局并实现 `Pod` —— 可以按原始字节直接上传.
///
/// 结构体大小为 80 字节, 是 16 的倍数 (WGSL 对 `mat4x4<f32>` 的对齐
/// 要求). **不要**删除 `_pad` —— 否则大小会变成 72, 不再满足对齐.
///
/// # 与 WGSL 的一致性
///
/// shader 中的 `Instance` struct **必须**与此处布局保持一致. **修改
/// 字段时同步修改 shader** —— 否则 shader 会按错误的 `stride` 读取
/// 实例数组.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct InstanceData {
	/// 从对象空间到世界空间的变换矩阵.
	pub model: [[f32; 4]; 4],

	/// 该实例的骨骼矩阵在 `BoneBuffer` 中的起始索引.
	///
	/// 非蒙皮实例填 0; 当 `bone_count == 0` 时该字段会被忽略.
	pub bone_offset: u32,

	/// 该实例使用的骨骼数量.
	///
	/// 取 0 表示没有骨骼 —— 着色器会跳过蒙皮计算, 直接用 `model` 变换
	/// 顶点.
	pub bone_count: u32,

	/// 显式的 `padding`, 让 `size_of::<InstanceData>()` 保持 16 的倍数.
	///
	/// **不要删除**. 部分后端要求 `storage buffer` 的 `stride` 按 16
	/// 字节对齐, 删掉后大小会变成 72.
	pub _pad: [u32; 2],
}

impl Default for InstanceData {
	fn default() -> Self {
		Self {
			model: Mat4::IDENTITY.to_cols_array_2d(),
			bone_offset: 0,
			bone_count: 0,
			_pad: [0; 2],
		}
	}
}

/// 存放实例数据的缓冲.
///
/// # 容量
///
/// 容量在创建时固定. 一旦超出, [`Self::push`] 与 [`Self::push_skinned`]
/// 都会返回 [`crate::system::Error::InstanceBufferFull`].
///
/// # 帧间复用
///
/// [`Self::clear`] 只重置使用计数, GPU 缓冲区保留. 因此跨帧复用时
/// 无需重新分配.
pub struct InstanceBuffer {
	buffer: wgpu::Buffer,
	bind_group: wgpu::BindGroup,
	bind_group_layout: wgpu::BindGroupLayout,
	capacity: u32,
	used: u32,
}

impl InstanceBuffer {
	/// 新建一个实例缓冲.
	///
	/// # 参数
	///
	/// - `device` —— 用于创建缓冲与 `bind group`.
	/// - `capacity` —— 最大实例数; 为 0 时提升到 1.
	pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
		let capacity = capacity.max(1);
		let stride = std::mem::size_of::<InstanceData>() as u64;

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
			label: Some("instance_buffer"),
			size,
			usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let bind_group_layout = device.create_bind_group_layout(
			&wgpu::BindGroupLayoutDescriptor {
				label: Some("instance_bind_group_layout"),
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
			},
		);

		let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("instance_bind_group"),
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

	// ================ 每帧状态 ================

	/// 把使用计数清零.
	///
	/// 每帧开始时调用一次, 让 [`Self::push`] 从索引 0 重新开始分配.
	pub fn clear(&mut self) {
		self.used = 0;
	}

	/// 本帧到目前为止使用的实例数.
	pub fn len(&self) -> u32 {
		self.used
	}

	/// 判断本帧是否尚未添加任何实例.
	pub fn is_empty(&self) -> bool {
		self.used == 0
	}

	/// 允许的最大实例数量.
	pub fn capacity(&self) -> u32 {
		self.capacity
	}

	// ================ 绑定资源 ================

	/// 供绑定使用的 `bind group`.
	pub fn bind_group(&self) -> &wgpu::BindGroup {
		&self.bind_group
	}

	/// 上述 `bind group` 的布局.
	pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
		&self.bind_group_layout
	}

	// ================ 实例写入 ================

	/// 添加一个不带骨骼的实例.
	///
	/// 返回该实例的索引 (从 0 开始).
	pub fn push(&mut self, queue: &wgpu::Queue, model: Mat4) -> Result<u32> {
		self.push_skinned(queue, model, 0, 0)
	}

	/// 添加一个携带骨骼信息的实例.
	///
	/// # 参数
	///
	/// - `model` —— 从对象空间到世界空间的变换矩阵.
	/// - `bone_offset` —— 这组骨骼在 [`crate::system::BoneBuffer`] 中的
	///   起始索引, 一般取自
	///   [`crate::system::BoneBuffer::push_skeleton`] 的返回值.
	/// - `bone_count` —— 骨骼数量; 取 0 表示没有骨骼.
	///
	/// # 返回
	///
	/// 该实例的索引 (从 0 开始).
	///
	/// # 错误
	///
	/// 容量用尽时返回 [`crate::system::Error::InstanceBufferFull`].
	pub fn push_skinned(
		&mut self,
		queue: &wgpu::Queue,
		model: Mat4,
		bone_offset: u32,
		bone_count: u32,
	) -> Result<u32> {
		if self.used >= self.capacity {
			return Err(crate::system::Error::InstanceBufferFull);
		}

		let offset = (self.used as u64) * std::mem::size_of::<InstanceData>() as u64;
		let data = InstanceData {
			model: model.to_cols_array_2d(),
			bone_offset,
			bone_count,
			_pad: [0; 2],
		};
		queue.write_buffer(&self.buffer, offset, bytemuck::bytes_of(&data));

		let index = self.used;
		self.used += 1;
		Ok(index)
	}
}
