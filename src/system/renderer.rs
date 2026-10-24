// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 渲染器 —— 持有长期资源, 把每帧拆成若干 pass 逐个记录.
//!
//! [`RenderSystem`] 持有渲染流程所需的全部长期资源. 渲染入口共三类:
//!
//! - [`RenderSystem::render_frame`] —— 只含单个 pass 的便捷封装.
//! - [`RenderSystem::begin_frame`] 配合 [`Frame::pass`] 与
//!   [`Frame::submit`] —— 容纳多个 pass.
//! - [`Frame::compute_pass`] —— 记录计算 pass.
//!
//! # 时间线查询
//!
//! 调用 [`RenderSystem::enable_timeline`] 之后, `Frame::pass` 与
//! `Frame::compute_pass` 会在 pass 前后自动写入 timestamp. 每帧结束
//! 时用 [`RenderSystem::read_timeline`] 读回各 pass 的毫秒耗时.
//!
//! 该功能要求设备支持 `Features::TIMESTAMP_QUERY`.
//!
//! # 调试标记
//!
//! [`PassHelper::push_debug_group`] 与 [`PassHelper::pop_debug_group`]
//! 会在 GPU 调试器 (`RenderDoc` / `PIX` / `Xcode` Metal Debugger) 中
//! 显示为分层事件. 后端不支持时静默忽略.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::pipeline::Target;
use crate::shader::Shader;
use crate::state::pipeline::{
	ComputePipelineDesc, PipelineCache, PipelineDesc,
};
use crate::state::{LayoutHandle, RenderType};
use crate::system::error::Result;
use crate::system::indices::SharedIndexBuffers;
use crate::system::mesh::GpuMesh;
use crate::system::objects::ObjectBuffer;
use crate::system::timeline::Timeline;
use crate::vertex::UploadedBuffer;

// ================ 全局 uniform 数据 ================

/// 帧级 uniform —— 绑定在 group 0 的 binding 0.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Globals {
	/// 相机的视图投影矩阵.
	pub view_projection: [[f32; 4]; 4],
	/// 视图矩阵 —— 只含视图部分, 不含投影.
	pub view: [[f32; 4]; 4],
	/// 由视图投影矩阵求逆得到的矩阵.
	pub inverse_view_projection: [[f32; 4]; 4],
	/// 相机的世界坐标; 第四个分量保留未用.
	pub camera_pos: [f32; 4],
	/// 雾的颜色 (RGB), 第四个分量不使用.
	pub fog_color: [f32; 4],
	/// 雾效的起始距离与结束距离.
	pub fog_range: [f32; 2],
	/// 游戏时间, 单位为秒.
	pub game_time: f32,
	/// 显式补齐字段 —— 使 `Globals` 的总大小为 16 的整数倍.
	pub _pad: f32,
}

impl Default for Globals {
	fn default() -> Self {
		Self {
			view_projection: glam::Mat4::IDENTITY.to_cols_array_2d(),
			view: glam::Mat4::IDENTITY.to_cols_array_2d(),
			inverse_view_projection: glam::Mat4::IDENTITY.to_cols_array_2d(),
			camera_pos: [0.0; 4],
			fog_color: [0.0, 0.0, 0.0, 1.0],
			fog_range: [32.0, 256.0],
			game_time: 0.0,
			_pad: 0.0,
		}
	}
}

// ================ 构造参数描述 ================

/// 创建 [`RenderSystem`] 时传入的参数.
pub struct RenderSystemDescriptor {
	/// 创建 GPU 资源所用的设备.
	pub device: Arc<wgpu::Device>,
	/// 提交命令与写入缓冲所用的队列.
	pub queue: Arc<wgpu::Queue>,
	/// 单个网格可用的最大顶点数.
	pub max_vertices: u32,
	/// 一帧内可提交的最大对象数.
	pub max_objects: u32,
}

// ================ 渲染系统 ================

/// 渲染系统 —— 每帧渲染流程的入口, 持有全部长期资源.
pub struct RenderSystem {
	device: Arc<wgpu::Device>,
	queue: Arc<wgpu::Queue>,
	pipeline_cache: PipelineCache,
	shared_indices: SharedIndexBuffers,
	object_buffer: ObjectBuffer,

	globals_buffer: wgpu::Buffer,
	globals_bind_group_layout: wgpu::BindGroupLayout,
	globals_bind_group: wgpu::BindGroup,

	globals: Globals,

	/// 可选的时间线; `None` 表示未启用.
	timeline: Option<Timeline>,
}

impl RenderSystem {
	/// 按描述创建渲染系统, 并初始化 globals 缓冲及其布局与绑定组.
	pub fn new(desc: RenderSystemDescriptor) -> Self {
		let globals_buffer = desc.device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("blaze3d_globals"),
			size: std::mem::size_of::<Globals>() as u64,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let globals_bind_group_layout =
			desc.device
				.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
					label: Some("blaze3d_globals_layout"),
					entries: &[wgpu::BindGroupLayoutEntry {
						binding: 0,
						visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
						ty: wgpu::BindingType::Buffer {
							ty: wgpu::BufferBindingType::Uniform,
							has_dynamic_offset: false,
							min_binding_size: None,
						},
						count: None,
					}],
				});

		let globals_bind_group = desc.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("blaze3d_globals_bind_group"),
			layout: &globals_bind_group_layout,
			entries: &[wgpu::BindGroupEntry {
				binding: 0,
				resource: globals_buffer.as_entire_binding(),
			}],
		});

		let pipeline_cache = PipelineCache::new(desc.device.clone());
		let shared_indices = SharedIndexBuffers::new(&desc.device, desc.max_vertices);
		let object_buffer = ObjectBuffer::new(&desc.device, desc.max_objects);

		Self {
			device: desc.device,
			queue: desc.queue,
			pipeline_cache,
			shared_indices,
			object_buffer,
			globals_buffer,
			globals_bind_group_layout,
			globals_bind_group,
			globals: Globals::default(),
			timeline: None,
		}
	}

	// ================ 资源访问器 ================

	/// 底层设备句柄.
	pub fn device(&self) -> &wgpu::Device {
		&self.device
	}

	/// 底层队列句柄.
	pub fn queue(&self) -> &wgpu::Queue {
		&self.queue
	}

	/// 读取帧级 uniform 的内容.
	pub fn globals(&self) -> &Globals {
		&self.globals
	}

	/// 修改帧级 uniform 的内容.
	///
	/// 改动会在下一次 `begin_frame` 时随 [`Globals`] 一起上传到 GPU.
	pub fn globals_mut(&mut self) -> &mut Globals {
		&mut self.globals
	}

	/// globals 对应的 bind group, 位于 group 0.
	pub fn globals_bind_group(&self) -> &wgpu::BindGroup {
		&self.globals_bind_group
	}

	/// globals 对应的 bind group layout, 位于 group 0.
	pub fn globals_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
		&self.globals_bind_group_layout
	}

	/// 共享索引缓冲 —— 供不带索引缓冲的网格按绘制模式复用.
	pub fn shared_indices(&self) -> &SharedIndexBuffers {
		&self.shared_indices
	}

	/// 以只读方式访问管线缓存.
	pub fn pipeline_cache(&self) -> &PipelineCache {
		&self.pipeline_cache
	}

	/// 以可变方式访问管线缓存.
	pub fn pipeline_cache_mut(&mut self) -> &mut PipelineCache {
		&mut self.pipeline_cache
	}

	/// 以只读方式访问对象缓冲.
	pub fn object_buffer(&self) -> &ObjectBuffer {
		&self.object_buffer
	}

	/// 以可变方式访问对象缓冲.
	pub fn object_buffer_mut(&mut self) -> &mut ObjectBuffer {
		&mut self.object_buffer
	}

	/// 对象 uniform 的 bind group, 位于 group 1.
	pub fn object_bind_group(&self) -> &wgpu::BindGroup {
		self.object_buffer.bind_group()
	}

	/// 对象 uniform 的 bind group layout, 位于 group 1.
	pub fn object_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
		self.object_buffer.bind_group_layout()
	}

	/// 以只读方式访问时间线; 返回 `None` 表示尚未启用.
	pub fn timeline(&self) -> Option<&Timeline> {
		self.timeline.as_ref()
	}

	// ================ 时间线开关 ================

	/// 启用 GPU 时间线.
	///
	/// # 行为
	///
	/// 内部创建一个 [`Timeline`]; 此后 `Frame::pass` 与
	/// `Frame::compute_pass` 会各自记录一对 timestamp.
	///
	/// # 参数
	///
	/// - `max_queries` —— 一帧最多能记录多少个 timestamp. 每个 pass
	///   占用两个, 因此通常传 `2 * max_passes`.
	///
	/// # 错误
	///
	/// 设备缺少 `TIMESTAMP_QUERY` 支持时, 返回
	/// [`crate::system::Error::TimestampQueryNotSupported`].
	pub fn enable_timeline(&mut self, max_queries: u32) -> Result<()> {
		self.timeline = Some(Timeline::new(&self.device, max_queries)?);
		Ok(())
	}

	/// 关闭时间线.
	///
	/// 已分配的 query set 与缓冲会随 `Timeline` 一并释放.
	pub fn disable_timeline(&mut self) {
		self.timeline = None;
	}

	/// 返回时间线当前是否处于启用状态.
	pub fn has_timeline(&self) -> bool {
		self.timeline.is_some()
	}

	/// 读回上一次 `submit` 所记录的时间线数据.
	///
	/// # 返回
	///
	/// 按记录顺序给出每个 pass 的 GPU 耗时, 单位为毫秒. 返回空 `Vec`
	/// 表示本帧没有记录到任何 pass.
	///
	/// # 阻塞
	///
	/// 内部要等 GPU 完成并映射缓冲 —— 会**阻塞当前线程**. 生产环境
	/// 建议每 N 帧读一次.
	///
	/// # 时间精度
	///
	/// 时间戳 tick 换算成纳秒的比例来自
	/// `Queue::get_timestamp_period()`. 各后端精度不同, 返回值已统一
	/// 换算为毫秒.
	///
	/// # 错误
	///
	/// - 时间线未启用 —— [`crate::system::Error::NoTimeline`].
	/// - 缓冲映射失败 —— [`crate::system::Error::TimelineMapFailed`].
	pub fn read_timeline(&self) -> Result<Vec<f32>> {
		use crate::system::Error;

		let timeline = self.timeline.as_ref().ok_or(Error::NoTimeline)?;

		if timeline.is_empty() {
			return Ok(Vec::new());
		}

		// 把时间戳 tick 换算成纳秒所需的系数.
		let timestamp_period = self.queue.get_timestamp_period();

		let slice = timeline.readback_buffer.slice(..);
		let (tx, rx) = std::sync::mpsc::channel();
		slice.map_async(wgpu::MapMode::Read, move |result| {
			let _ = tx.send(result);
		});

		self.device
			.poll(wgpu::PollType::Wait {
				submission_index: None,
				timeout: None,
			})
			.map_err(|_| Error::TimelineMapFailed)?;

		rx.recv()
			.map_err(|_| Error::TimelineMapFailed)?
			.map_err(|_| Error::TimelineMapFailed)?;

		let mapped = slice
			.get_mapped_range()
			.map_err(|_| Error::TimelineMapFailed)?;

		let values: &[u64] = bytemuck::cast_slice(&mapped);

		let mut result = Vec::with_capacity(timeline.used_ranges.len());
		for &(begin, end) in &timeline.used_ranges {
			let begin_v = values[begin as usize];
			let end_v = values[end as usize];
			let duration_ticks = end_v.saturating_sub(begin_v);
			let duration_ns = (duration_ticks as f32) * timestamp_period;
			let duration_ms = duration_ns / 1_000_000.0;
			result.push(duration_ms);
		}

		drop(mapped);
		timeline.readback_buffer.unmap();

		Ok(result)
	}

	/// 上传单个对象的 model 矩阵, 返回它在对象缓冲中的 dynamic offset.
	pub fn push_object(&mut self, model: glam::Mat4) -> Result<u32> {
		self.object_buffer.push(&self.queue, model)
	}

	// ================ 帧的开始与提交 ================

	/// 开启新的一帧.
	///
	/// # 行为
	///
	/// 1. 把 [`ObjectBuffer`] 的使用计数清零.
	/// 2. 重置时间线中记录的帧状态.
	/// 3. 把当前的 [`Globals`] 写入 GPU 缓冲.
	/// 4. 新建一个 command encoder.
	/// 5. 返回可继续记录 pass 的 [`Frame`].
	pub fn begin_frame(&mut self) -> Frame<'_> {
		self.object_buffer.clear();

		if let Some(timeline) = self.timeline.as_mut() {
			timeline.reset();
		}

		self.queue.write_buffer(
			&self.globals_buffer,
			0,
			bytemuck::bytes_of(&self.globals),
		);

		Frame {
			encoder: self
				.device
				.create_command_encoder(&wgpu::CommandEncoderDescriptor {
					label: Some("blaze3d_frame"),
				}),
			system: self,
		}
	}

	/// 只渲染单个 pass 的一帧.
	///
	/// 依次调用 [`Self::begin_frame`], [`Frame::pass`] 与
	/// [`Frame::submit`], 并原样返回回调的返回值.
	pub fn render_frame<T, F, R>(&mut self, target: &T, f: F) -> R
	where
		T: Target,
		F: FnOnce(&mut PassHelper<'_, '_>) -> R,
	{
		let mut frame = self.begin_frame();
		let result = frame.pass(target, f);
		frame.submit();
		result
	}
}

// ================ 帧 ================

/// 一帧的渲染上下文 —— 持有 command encoder 与渲染系统的可变借用.
pub struct Frame<'a> {
	encoder: wgpu::CommandEncoder,
	system: &'a mut RenderSystem,
}

impl<'a> Frame<'a> {
	/// 在帧内记录一个 render pass.
	///
	/// # 时间线
	///
	/// 时间线启用时, 本方法会在 pass 前后各写一个 timestamp; 该 pass
	/// 的耗时随后由 [`RenderSystem::read_timeline`] 读回.
	///
	/// # 调试标记
	///
	/// 本方法**不会**自动 push 调试组. 若需要在 `RenderDoc` / `PIX`
	/// 里看到分层结构, 请在回调中成对调用
	/// [`PassHelper::push_debug_group`] 与
	/// [`PassHelper::pop_debug_group`].
	pub fn pass<T, F, R>(&mut self, target: &T, f: F) -> R
	where
		T: Target,
		F: FnOnce(&mut PassHelper<'_, '_>) -> R,
	{
		// 时间线启用时先占一对 timestamp 索引.
		let pair = self
			.system
			.timeline
			.as_mut()
			.and_then(|t| t.allocate_pair());

		// 写入起始时间戳.
		if let Some((begin, _)) = pair {
			let query_set = &self.system.timeline.as_ref().unwrap().query_set;
			self.encoder.write_timestamp(query_set, begin);
		}

		// 记录 pass 本身.
		let result = {
			let mut pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("blaze3d_pass"),
				color_attachments: &[Some(target.color_attachment())],
				depth_stencil_attachment: target.depth_attachment(),
				timestamp_writes: None,
				occlusion_query_set: None,
				multiview_mask: None,
			});

			let RenderSystem {
				pipeline_cache,
				shared_indices,
				object_buffer,
				globals_bind_group,
				..
			} = &mut *self.system;

			let mut helper = PassHelper {
				pass: &mut pass,
				pipeline_cache,
				shared_indices,
				globals_bind_group,
				object_bind_group: object_buffer.bind_group(),
				current_pipeline_desc: None,
			};

			let (viewport_w, viewport_h) = target.viewport_size();
			helper.set_viewport(0.0, 0.0, viewport_w as f32, viewport_h as f32);

			f(&mut helper)
		};

		// 写入结束时间戳.
		if let Some((_, end)) = pair {
			let query_set = &self.system.timeline.as_ref().unwrap().query_set;
			self.encoder.write_timestamp(query_set, end);
		}

		result
	}

	/// 在帧内记录一个计算 pass.
	///
	/// # 时间线
	///
	/// 行为与 [`Self::pass`] 一致 —— 同样自动写入前后两个 timestamp.
	pub fn compute_pass<F>(&mut self, f: F)
	where
		F: FnOnce(&mut ComputePassHelper<'_, '_>),
	{
		let pair = self
			.system
			.timeline
			.as_mut()
			.and_then(|t| t.allocate_pair());

		if let Some((begin, _)) = pair {
			let query_set = &self.system.timeline.as_ref().unwrap().query_set;
			self.encoder.write_timestamp(query_set, begin);
		}

		{
			let mut pass = self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
				label: Some("blaze3d_compute_pass"),
				timestamp_writes: None,
			});

			let RenderSystem { pipeline_cache, .. } = &mut *self.system;

			let mut helper = ComputePassHelper {
				pass: &mut pass,
				pipeline_cache,
				current_pipeline_desc: None,
			};

			f(&mut helper);
		}

		if let Some((_, end)) = pair {
			let query_set = &self.system.timeline.as_ref().unwrap().query_set;
			self.encoder.write_timestamp(query_set, end);
		}
	}

	/// 把单个对象的 model 矩阵写入对象缓冲, 返回 dynamic offset.
	pub fn push_object(&mut self, model: glam::Mat4) -> Result<u32> {
		self.system.object_buffer.push(&self.system.queue, model)
	}

	/// 结束并提交本帧.
	///
	/// # 时间线
	///
	/// 时间线已启用且本帧记录过 pass 时, 会先 `resolve_query_set` 并
	/// 把结果拷贝到 readback buffer, 之后可用
	/// 用 [`RenderSystem::read_timeline`] 读取.
	pub fn submit(mut self) {
		// 处理时间线: resolve query 结果并搬到读回缓冲.
		if let Some(timeline) = self.system.timeline.as_ref()
			&& !timeline.is_empty()
		{
			let count = timeline.next_query;
			let byte_size = (count as u64) * 8;

			self.encoder.resolve_query_set(
				&timeline.query_set,
				0..count,
				&timeline.resolve_buffer,
				0,
			);
			self.encoder.copy_buffer_to_buffer(
				&timeline.resolve_buffer,
				0,
				&timeline.readback_buffer,
				0,
				byte_size,
			);
		}

		let command_buffer = self.encoder.finish();
		self.system.queue.submit(Some(command_buffer));
	}
}

// ================ pass 绘制接口 ================

/// 包装 `wgpu::RenderPass`, 提供绘制所需的便捷方法.
pub struct PassHelper<'a, 'b> {
	pass: &'a mut wgpu::RenderPass<'b>,
	pipeline_cache: &'a mut PipelineCache,
	shared_indices: &'a SharedIndexBuffers,
	globals_bind_group: &'a wgpu::BindGroup,
	object_bind_group: &'a wgpu::BindGroup,

	current_pipeline_desc: Option<PipelineDesc>,
}

impl<'a, 'b> PassHelper<'a, 'b> {
	// ================ 底层 pass 访问 ================

	/// 返回底层 [`wgpu::RenderPass`] 的可变引用, 便于调用未包装的接口.
	pub fn raw(&mut self) -> &mut wgpu::RenderPass<'b> {
		self.pass
	}

	// ================ 渲染 pass 调试标记 ================

	/// 在 GPU 调试器中开启一个带标签的组.
	///
	/// # 行为
	///
	/// 在 GPU 调试器 (`RenderDoc` / `PIX` / `Xcode`) 中显示为嵌套
	/// 事件. 后端不支持时**静默忽略**, 不影响渲染结果.
	///
	/// # 配对
	///
	/// 每个 `push_debug_group` 都应有对应的 `pop_debug_group`. 缺少
	/// 配对不会报错, 但会让调试器中的事件层次错乱.
	///
	/// # 用法
	///
	/// ```ignore
	/// frame.pass(&target, |pass| {
	///     pass.push_debug_group("terrain");
	///     pass.set_render_type(&rt, &shader, &layout);
	///     // ... 绘制地形 ...
	///     pass.pop_debug_group();
	///
	///     pass.push_debug_group("entities");
	///     // ... 绘制实体 ...
	///     pass.pop_debug_group();
	/// });
	/// ```
	pub fn push_debug_group(&mut self, label: &str) {
		self.pass.push_debug_group(label);
	}

	/// 结束最近一个尚未结束的调试组.
	pub fn pop_debug_group(&mut self) {
		self.pass.pop_debug_group();
	}

	// ================ 视口设置 ================

	/// 设置视口的矩形区域, 深度范围固定为 `0.0` 到 `1.0`.
	pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32) {
		self.pass.set_viewport(x, y, w, h, 0.0, 1.0);
	}

	/// 设置视口的矩形区域, 并显式给出最小与最大深度.
	pub fn set_viewport_with_depth(
		&mut self,
		x: f32,
		y: f32,
		w: f32,
		h: f32,
		min_depth: f32,
		max_depth: f32,
	) {
		self.pass.set_viewport(x, y, w, h, min_depth, max_depth);
	}

	// ================ 渲染管线 ================

	/// 设置渲染管线, 并绑定 group 0 的 globals.
	///
	/// 管线描述与当前管线相同时直接返回 —— 连续的同类绘制不会重复
	/// 调用 `set_pipeline`.
	pub fn set_render_type(
		&mut self,
		rt: &RenderType,
		shader: &Shader,
		layout: &LayoutHandle,
	) {
		let desc = PipelineDesc {
			render_type: rt.clone(),
			shader_id: shader.id,
			layout_id: layout.id(),
		};

		if self.current_pipeline_desc.as_ref() == Some(&desc) {
			return;
		}

		let pipeline = self
			.pipeline_cache
			.get_or_create(&desc, &shader.module, layout.raw());

		self.pass.set_pipeline(&pipeline);
		self.pass.set_bind_group(0, self.globals_bind_group, &[]);

		self.current_pipeline_desc = Some(desc);
	}

	/// 用 dynamic offset 绑定 group 1 的对象 uniform.
	pub fn set_object(&mut self, offset: u32) {
		self.pass
			.set_bind_group(1, self.object_bind_group, &[offset]);
	}

	// ================ 网格绘制 ================

	/// 绘制常驻 GPU 的网格; 网格自带索引缓冲时走 indexed 绘制.
	pub fn draw_mesh(&mut self, mesh: &GpuMesh) {
		self.pass
			.set_vertex_buffer(0, mesh.vertex_buffer().slice(..));

		match mesh.index_buffer() {
			Some(idx_buf) => {
				self.pass
					.set_index_buffer(idx_buf.slice(..), wgpu::IndexFormat::Uint32);
				self.pass.draw_indexed(0..mesh.index_count(), 0, 0..1);
			}
			None => {
				if let Some((shared_buf, _)) = self.shared_indices.buffer_for(mesh.mode()) {
					self.pass
						.set_index_buffer(shared_buf.slice(..), wgpu::IndexFormat::Uint32);
					self.pass.draw_indexed(0..mesh.index_count(), 0, 0..1);
				} else {
					self.pass.draw(0..mesh.vertex_count(), 0..1);
				}
			}
		}
	}

	/// 以实例化方式绘制常驻 GPU 的网格; 实例数为 0 时直接返回.
	pub fn draw_mesh_instanced(
		&mut self,
		mesh: &GpuMesh,
		instances: &crate::system::InstanceBuffer,
	) {
		let instance_count = instances.len();
		if instance_count == 0 {
			return;
		}

		self.pass.set_bind_group(1, instances.bind_group(), &[]);
		self.pass
			.set_vertex_buffer(0, mesh.vertex_buffer().slice(..));

		let instance_range = 0..instance_count;

		match mesh.index_buffer() {
			Some(idx_buf) => {
				self.pass
					.set_index_buffer(idx_buf.slice(..), wgpu::IndexFormat::Uint32);
				self.pass
					.draw_indexed(0..mesh.index_count(), 0, instance_range);
			}
			None => {
				if let Some((shared_buf, _)) = self.shared_indices.buffer_for(mesh.mode()) {
					self.pass
						.set_index_buffer(shared_buf.slice(..), wgpu::IndexFormat::Uint32);
					self.pass
						.draw_indexed(0..mesh.index_count(), 0, instance_range);
				} else {
					self.pass.draw(0..mesh.vertex_count(), instance_range);
				}
			}
		}
	}

	/// 以实例化方式绘制带骨骼的网格.
	///
	/// 除 group 1 的实例数据外, 还额外把骨骼矩阵绑到 group 2.
	pub fn draw_mesh_instanced_skinned(
		&mut self,
		mesh: &GpuMesh,
		instances: &crate::system::InstanceBuffer,
		bones: &crate::system::BoneBuffer,
	) {
		let instance_count = instances.len();
		if instance_count == 0 {
			return;
		}

		self.pass.set_bind_group(1, instances.bind_group(), &[]);
		self.pass.set_bind_group(2, bones.bind_group(), &[]);
		self.pass
			.set_vertex_buffer(0, mesh.vertex_buffer().slice(..));

		let instance_range = 0..instance_count;

		match mesh.index_buffer() {
			Some(idx_buf) => {
				self.pass
					.set_index_buffer(idx_buf.slice(..), wgpu::IndexFormat::Uint32);
				self.pass
					.draw_indexed(0..mesh.index_count(), 0, instance_range);
			}
			None => {
				if let Some((shared_buf, _)) = self.shared_indices.buffer_for(mesh.mode()) {
					self.pass
						.set_index_buffer(shared_buf.slice(..), wgpu::IndexFormat::Uint32);
					self.pass
						.draw_indexed(0..mesh.index_count(), 0, instance_range);
				} else {
					self.pass.draw(0..mesh.vertex_count(), instance_range);
				}
			}
		}
	}

	/// 绘制一次性的 [`UploadedBuffer`] 数据.
	pub fn draw_uploaded(&mut self, uploaded: &UploadedBuffer) {
		let state = &uploaded.draw_state;

		let Some(vertex_buffer) = &uploaded.vertex_buffer else {
			return;
		};

		self.pass.set_vertex_buffer(0, vertex_buffer.slice(..));

		if let Some(index_buffer) = &uploaded.index_buffer {
			self.pass
				.set_index_buffer(index_buffer.slice(..), state.index_type.wgpu());
			self.pass.draw_indexed(0..state.index_count, 0, 0..1);
			return;
		}

		if let Some((shared_buffer, _)) = self.shared_indices.buffer_for(state.mode) {
			self.pass
				.set_index_buffer(shared_buffer.slice(..), wgpu::IndexFormat::Uint32);
			self.pass.draw_indexed(0..state.index_count, 0, 0..1);
			return;
		}

		self.pass.draw(0..state.vertex_count, 0..1);
	}

	// ================ bind group 与绘制透传 ================

	/// 绑定 group 2 及以后的 bind group.
	pub fn set_bind_group(
		&mut self,
		index: u32,
		bind_group: &wgpu::BindGroup,
		offsets: &[wgpu::DynamicOffset],
	) {
		self.pass.set_bind_group(index, bind_group, offsets);
	}

	/// 直接转调 [`wgpu::RenderPass::draw`].
	pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
		self.pass.draw(vertices, instances);
	}

	/// 直接转调 [`wgpu::RenderPass::draw_indexed`].
	pub fn draw_indexed(
		&mut self,
		indices: std::ops::Range<u32>,
		base_vertex: i32,
		instances: std::ops::Range<u32>,
	) {
		self.pass.draw_indexed(indices, base_vertex, instances);
	}
}

// ================ 计算 pass 接口 ================

/// 包装 `wgpu::ComputePass`, 提供 dispatch 所需的便捷方法.
pub struct ComputePassHelper<'a, 'b> {
	pass: &'a mut wgpu::ComputePass<'b>,
	pipeline_cache: &'a mut PipelineCache,

	current_pipeline_desc: Option<ComputePipelineDesc>,
}

impl<'a, 'b> ComputePassHelper<'a, 'b> {
	// ================ 底层计算 pass 访问 ================

	/// 返回底层 [`wgpu::ComputePass`] 的可变引用, 便于调用未包装的接口.
	pub fn raw(&mut self) -> &mut wgpu::ComputePass<'b> {
		self.pass
	}

	// ================ 计算 pass 调试标记 ================

	/// 在计算 pass 中开启一个调试组.
	///
	/// 标签即调试器中该事件的名字; 后端不支持时静默忽略.
	pub fn push_debug_group(&mut self, label: &str) {
		self.pass.push_debug_group(label);
	}

	/// 结束最近打开的那个调试组.
	pub fn pop_debug_group(&mut self) {
		self.pass.pop_debug_group();
	}

	// ================ 计算管线 ================

	/// 切换到给定着色器与布局对应的计算管线.
	///
	/// 管线描述与当前管线相同时直接返回, 不会重复调用 `set_pipeline`.
	pub fn set_compute_type(&mut self, shader: &Shader, layout: &LayoutHandle) {
		let desc = ComputePipelineDesc {
			shader_id: shader.id,
			layout_id: layout.id(),
		};

		if self.current_pipeline_desc.as_ref() == Some(&desc) {
			return;
		}

		let pipeline = self.pipeline_cache.get_or_create_compute(
			&desc,
			&shader.module,
			layout.raw(),
		);

		self.pass.set_pipeline(&pipeline);
		self.current_pipeline_desc = Some(desc);
	}

	// ================ 计算 pass 绑定 ================

	/// 绑定指定索引的 bind group, 并给出动态偏移.
	pub fn set_bind_group(
		&mut self,
		index: u32,
		bind_group: &wgpu::BindGroup,
		offsets: &[wgpu::DynamicOffset],
	) {
		self.pass.set_bind_group(index, bind_group, offsets);
	}

	// ================ 分发 ================

	/// 直接指定三个维度的 workgroup 数量.
	pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
		self.pass.dispatch_workgroups(x, y, z);
	}

	/// 从缓冲中读取 workgroup 数量, 即间接分发.
	pub fn dispatch_indirect(&mut self, buffer: &wgpu::Buffer, offset: u64) {
		self.pass.dispatch_workgroups_indirect(buffer, offset);
	}
}
