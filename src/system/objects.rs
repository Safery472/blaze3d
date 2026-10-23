//! 每对象 `uniform` 缓冲.
//!
//! [`ObjectBuffer`] 是一段连续的 `uniform` 缓冲, 每次绘制时通过
//! `dynamic offset` 选中其中一个对象的数据. 与「每个对象一个缓冲」的
//! 做法相比, 它的特点如下:
//!
//! - **一次绑定, 多次绘制** —— `bind group` 只创建一次, 而
//!   `dynamic offset` 在 `set_bind_group` 时指定.
//! - **内存紧凑** —— 所有对象的数据连续存放.
//! - **对齐约束** —— 每个对象的起始偏移必须满足设备给出的对齐要求
//!   (即 [`wgpu::Limits::min_uniform_buffer_offset_alignment`]).
//!
//! # 与 Java 版的差异
//!
//! Java 版的「每物体数据」分散在各 shader 的 `uniform` 变量中, 没有
//! 统一缓冲. Rust 版改用一段连续缓冲加 `dynamic offset` 表达 —— 这既
//! 是 wgpu 中绑定多个对象的惯用法, 也正是 Java 版想要达到却未显式
//! 实现的效果.
//!
//! # 对齐
//!
//! `stride` 取 `min_uniform_buffer_offset_alignment` 与 256 中的较大
//! 值. 后者是常见硬件的最小值, 显式取 `max` 可以避免驱动报告的
//! 异常值造成对齐错误.
//!
//! `ObjectUniforms` 通常小于 `stride` (例如 64 字节对 256 字节), 因此
//! 缓冲中每个对象之间都有浪费 —— 这是 `dynamic offset` 机制无法避免
//! 的代价.
//!
//! # 使用流程
//!
//! 1. 帧开始时调用 [`ObjectBuffer::clear`] 把对象计数清零.
//! 2. 为每个可见对象调用 [`ObjectBuffer::push`], 拿到对应的 `offset`.
//! 3. 绘制时把该 `offset` 交给
//!    [`crate::system::PassHelper::set_object`].

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use crate::system::error::{Error, Result};

/// 单个对象的 `uniform` 数据.
///
/// 目前只包含 `model` 矩阵. 将来要扩展材质参数, 颜色调整或光照因子
/// 时, 直接往这里加字段即可 —— 但**必须保持 `Pod`** (无 `padding`,
/// 无 `Drop`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ObjectUniforms {
	/// 从对象空间到世界空间的变换矩阵.
	pub model: [[f32; 4]; 4],
}

impl Default for ObjectUniforms {
	fn default() -> Self {
		Self {
			model: Mat4::IDENTITY.to_cols_array_2d(),
		}
	}
}

/// 每对象 `uniform` 的环形缓冲.
///
/// # 环形
///
/// 缓冲大小由 `capacity` 在创建时固定. 每次 [`Self::push`] 分配下一个
/// 槽位, `used` 随之递增; 帧开始时 [`Self::clear`] 把 `used` 重置为 0
/// —— 同一段缓冲便能跨帧复用, 无需重新分配.
///
/// 这并非严格的「环形」: 一帧内单调增长, 帧间才重置. 名字沿用了通用
/// 术语, 但实际语义只是「帧内环形」.
///
/// # 与 `bind group` 的关系
///
/// `bind group` 只创建一次, 绑定整个缓冲, 但把 `size` 限定为单个
/// `ObjectUniforms`. 具体使用哪个槽位由绘制时的 `dynamic offset` 选择.
pub struct ObjectBuffer {
	buffer: wgpu::Buffer,
	bind_group: wgpu::BindGroup,
	bind_group_layout: wgpu::BindGroupLayout,

	/// 每个对象占用的字节步长, 已按设备对齐要求取整.
	stride: u32,

	/// 每帧允许的最多对象数.
	capacity: u32,

	/// 本帧到目前为止占用的槽位数.
	used: u32,
}

impl ObjectBuffer {
	/// 新建一个对象缓冲.
	///
	/// # 参数
	///
	/// - `device` —— 用于创建缓冲与 `bind group`.
	/// - `capacity` —— 每帧的最大对象数; 传 0 会被提升到 1, 因为 wgpu
	///   拒绝空缓冲.
	///
	/// # 对齐
	///
	/// `stride` 取 `min_uniform_buffer_offset_alignment` 与 256 中的较大
	/// 值. 前者是驱动报告的最小对齐, 后者是常见硬件的实际取值, 取
	/// `max` 可以同时满足两者.
	pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
		let capacity = capacity.max(1);

		// `dynamic offset` 必须对齐到
		// `min_uniform_buffer_offset_alignment`. 再与 256 取 `max` 是
		// 保险措施 —— 部分驱动只报告 64, 实际却要求 256.
		let align = device
			.limits()
			.min_uniform_buffer_offset_alignment
			.max(256);
		let stride = align;
		let total_size = (capacity as u64) * (stride as u64);

		let buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("object_uniforms"),
			size: total_size,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let bind_group_layout = device.create_bind_group_layout(
			&wgpu::BindGroupLayoutDescriptor {
				label: Some("object_bind_group_layout"),
				entries: &[wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::VERTEX,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Uniform,
						// `dynamic offset` 让同一个 `bind group` 能按 `offset`
						// 切换到另一段数据.
						has_dynamic_offset: true,
						// 把数据大小限定为单个对象 —— wgpu 用它做边界检查.
						min_binding_size: wgpu::BufferSize::new(
							size_of::<ObjectUniforms>() as u64,
						),
					},
					count: None,
				}],
			},
		);

		let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("object_bind_group"),
			layout: &bind_group_layout,
			entries: &[wgpu::BindGroupEntry {
				binding: 0,
				resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
					buffer: &buffer,
					offset: 0,
					// `size` 限定为单个 `ObjectUniforms` —— `offset` 由绘制
					// 时的 `dynamic offset` 提供.
					size: wgpu::BufferSize::new(
						size_of::<ObjectUniforms>() as u64,
					),
				}),
			}],
		});

		Self {
			buffer,
			bind_group,
			bind_group_layout,
			stride,
			capacity,
			used: 0,
		}
	}

	// ================ 帧状态 ================

	/// 把使用计数清零.
	///
	/// 每帧开始时调用一次, 让 [`Self::push`] 从槽位 0 重新开始分配.
	///
	/// # 时序
	///
	/// 该调用**必须**发生在
	/// [`crate::system::RenderSystem::render_frame`] 之前 ——
	/// `render_frame` 内部会调用它; 但若调用方希望在渲染之前预上传
	/// 对象 (即在 `render_frame` 之外调用 `push`), 就必须自己先
	/// `clear` 一次.
	///
	/// 常规用法依靠 `render_frame` 内部调用, 调用方无需手动处理.
	pub fn clear(&mut self) {
		self.used = 0;
	}

	/// 每帧允许的最多对象数.
	pub fn capacity(&self) -> u32 {
		self.capacity
	}

	/// 本帧到目前为止占用的槽位数.
	pub fn used(&self) -> u32 {
		self.used
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

	// ================ 写入缓冲 ================

	/// 上传单个对象的 `model` 矩阵, 返回 `dynamic offset`.
	///
	/// # 错误
	///
	/// 缓冲已满时返回 [`Error::ObjectBufferFull`], 应对方式见该错误的文档.
	///
	/// # 返回值
	///
	/// 返回字节偏移, 供 [`crate::system::PassHelper::set_object`] 使用.
	/// 该偏移在帧内有效, 帧结束 (即 `render_frame` 完成) 后失效.
	pub fn push(&mut self, queue: &wgpu::Queue, model: Mat4) -> Result<u32> {
		self.push_uniforms(queue, ObjectUniforms { model: model.to_cols_array_2d() })
	}

	/// 写入一份完整的 [`ObjectUniforms`].
	///
	/// 与 [`Self::push`] 的区别在于: 它允许设置 `model` 之外的字段
	/// (留给将来扩展材质参数时使用).
	pub fn push_uniforms(
		&mut self,
		queue: &wgpu::Queue,
		uniforms: ObjectUniforms,
	) -> Result<u32> {
		if self.used >= self.capacity {
			return Err(Error::ObjectBufferFull);
		}

		let offset = self.used * self.stride;
		queue.write_buffer(&self.buffer, offset as u64, bytemuck::bytes_of(&uniforms));

		self.used += 1;
		Ok(offset)
	}
}
