//! GPU 时间线 —— 记录并读回每个 pass 的耗时.
//!
//! [`Timeline`] 借助 wgpu 的 `QuerySet` 统计每个 pass 的 GPU 耗时.
//! 时间线启用后, [`crate::system::Frame::pass`] 与
//! [`crate::system::Frame::compute_pass`] 会在 pass 前后自动写入
//! timestamp.
//!
//! # 特性要求
//!
//! 需要设备提供**两个** feature:
//!
//! - [`wgpu::Features::TIMESTAMP_QUERY`] —— 供 pass 内部写入.
//! - [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`] —— 供
//!   **encoder 级别**写入 (`CommandEncoder::write_timestamp`).
//!
//! 本模块走的是第二种 —— 在 pass 之外记录起止时间. 若只请求了
//! `TIMESTAMP_QUERY`, 调用 `write_timestamp` 会在运行时 panic.
//!
//! # 时间精度
//!
//! 时间戳的单位由 `Queue::get_timestamp_period()` 给出, 即每个 tick
//! 对应多少纳秒; 各后端精度不同. 对外返回的结果已换算成毫秒.
//!
//! # 读取成本
//!
//! `read_timeline` 内部要等 GPU 完成并映射缓冲 —— 会**阻塞当前线程**.
//! 生产环境应当每 N 帧读一次, 而不是每帧都读.

use wgpu::BufferUsages;

use super::error::{Error, Result};

/// 时间线要求的 feature 组合.
///
/// 两个 feature 缺一不可, 原因见模块文档.
pub(crate) const TIMELINE_FEATURES: wgpu::Features = wgpu::Features::TIMESTAMP_QUERY
	.union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

/// GPU 时间线 —— 每个 pass 一对 timestamp, 据此算出耗时.
pub struct Timeline {
	pub(crate) query_set: wgpu::QuerySet,
	pub(crate) resolve_buffer: wgpu::Buffer,
	pub(crate) readback_buffer: wgpu::Buffer,
	pub(crate) max_queries: u32,

	/// 本帧已经分配出去的 query 数量.
	pub(crate) next_query: u32,

	/// 本帧用到的 (begin, end) query 索引对.
	pub(crate) used_ranges: Vec<(u32, u32)>,
}

impl Timeline {
	/// 创建一条时间线.
	///
	/// # 参数
	///
	/// - `device` —— 用于创建 query set 与各个缓冲.
	/// - `max_queries` —— query 数量上限; 每个 pass 占用两个, 传小于 2
	///   的值会被抬到 2.
	///
	/// # 错误
	///
	/// 设备不能同时支持 `TIMESTAMP_QUERY` 与
	/// `TIMESTAMP_QUERY_INSIDE_ENCODERS` 时, 返回
	/// [`Error::TimestampQueryNotSupported`].
	pub fn new(device: &wgpu::Device, max_queries: u32) -> Result<Self> {
		if !device.features().contains(TIMELINE_FEATURES) {
			return Err(Error::TimestampQueryNotSupported);
		}

		let max_queries = max_queries.max(2);
		let byte_size = (max_queries as u64) * 8;

		let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
			label: Some("timeline_query_set"),
			ty: wgpu::QueryType::Timestamp,
			count: max_queries,
		});

		let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("timeline_resolve_buffer"),
			size: byte_size,
			usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
			mapped_at_creation: false,
		});

		let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("timeline_readback_buffer"),
			size: byte_size,
			usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
			mapped_at_creation: false,
		});

		Ok(Self {
			query_set,
			resolve_buffer,
			readback_buffer,
			max_queries,
			next_query: 0,
			used_ranges: Vec::new(),
		})
	}

	/// 复位本帧的计数与已用索引对.
	pub(crate) fn reset(&mut self) {
		self.next_query = 0;
		self.used_ranges.clear();
	}

	/// 为一次计时配一对 query 索引 `(begin, end)`.
	///
	/// 本帧的 query 已经用完时返回 `None`.
	pub(crate) fn allocate_pair(&mut self) -> Option<(u32, u32)> {
		if self.next_query + 2 > self.max_queries {
			return None;
		}
		let begin = self.next_query;
		let end = begin + 1;
		self.next_query += 2;
		self.used_ranges.push((begin, end));
		Some((begin, end))
	}

	/// 返回本帧已记录的 pass 数量.
	pub fn pass_count(&self) -> usize {
		self.used_ranges.len()
	}

	/// 返回本帧是否还没有任何记录.
	pub fn is_empty(&self) -> bool {
		self.used_ranges.is_empty()
	}

	/// 返回容量, 即一帧最多能容纳多少个 query.
	pub fn max_queries(&self) -> u32 {
		self.max_queries
	}

	/// 返回容量换算出的 pass 上限, 即 `max_queries / 2`.
	pub fn max_passes(&self) -> u32 {
		self.max_queries / 2
	}
}
