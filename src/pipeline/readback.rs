//! 纹理与缓冲读回.
//!
//! 把 GPU 上的资源复制回 CPU, 用于截图, 测试断言, GPU 计算结果验证.
//!
//! # 同步语义
//!
//! [`read_texture`] / [`read_buffer`] 在内部提交拷贝命令并等待完成,
//! 调用方无需手动同步. 这是有意为之: 读回的典型场景是「渲染完一帧后
//! 读取结果」, 同步等待比让调用方管理 future 更简单.
//!
//! 需要异步读回的场景 (例如多帧流水线) 应直接使用 wgpu 原生的
//! `map_async`, 不要经过本模块.
//!
//! # 支持范围
//!
//! - **纹理**: 只支持未压缩的 2D 纹理; 压缩格式的块尺寸不是 1x1, 与
//!   「每像素字节数」的语义不符.
//! - **缓冲**: 任何带 `COPY_SRC` 用途的缓冲, 由调用方指定要读取的字节数.
//!
//! # 与 `wgpu::Queue::read_buffer` 的关系
//!
//! wgpu 30 起提供 `Queue::read_buffer`, 但它只接受 buffer, 且语义与
//! 本模块的 [`read_buffer`] 略有不同: 它读取 buffer 中已有的内容, 不做
//! 拷贝. 本模块的 [`read_buffer`] 显式创建 staging buffer 并提交拷贝,
//! 因此适用于任何带 `COPY_SRC` 用途的 buffer.

use super::error::{Error, Result};
use super::target::RenderTarget;

/// 从纹理读回像素数据.
///
/// # 前置条件
///
/// - 纹理带 `COPY_SRC` 用途.
/// - `width` / `height` 是纹理 mip 0 的实际尺寸.
/// - `format` 与纹理创建时的格式一致.
/// - 格式是未压缩的 2D 格式.
///
/// # 返回值
///
/// 紧密排列的像素字节. 每个像素占 `block_copy_size` 字节, 行间没有
/// padding.
///
/// # 错误
///
/// - 尺寸为 0: [`Error::ZeroSize`].
/// - 格式不支持: [`Error::ReadbackUnsupportedFormat`].
/// - 缓冲映射失败: [`Error::ReadbackMapFailed`].
/// - 等待 GPU 完成失败: [`Error::ReadbackPollFailed`].
///
/// # 同步语义
///
/// 内部提交一次拷贝命令并等待完成, 阻塞当前线程直到 GPU 完成. 多帧
/// 流水线场景应直接使用 wgpu 原生的异步接口.
pub fn read_texture(
	device: &wgpu::Device,
	queue: &wgpu::Queue,
	texture: &wgpu::Texture,
	width: u32,
	height: u32,
	format: wgpu::TextureFormat,
) -> Result<Vec<u8>> {
	if width == 0 || height == 0 {
		return Err(Error::ZeroSize { width, height });
	}

	let bytes_per_pixel = bytes_per_pixel(format)?;

	// wgpu 要求 copy_texture_to_buffer 的 bytes_per_row 对齐到
	// COPY_BYTES_PER_ROW_ALIGNMENT (256 字节), 或者一次 copy 整个 mip.
	// 前者更通用.
	let unpadded_bytes_per_row = width * bytes_per_pixel;
	let padded_bytes_per_row =
		align_to(unpadded_bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

	let buffer_size = padded_bytes_per_row as u64 * height as u64;

	let buffer = device.create_buffer(&wgpu::BufferDescriptor {
		label: Some("blaze3d_readback_buffer"),
		size: buffer_size,
		usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
		mapped_at_creation: false,
	});

	let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
		label: Some("blaze3d_readback_encoder"),
	});

	encoder.copy_texture_to_buffer(
		wgpu::TexelCopyTextureInfo {
			texture,
			mip_level: 0,
			origin: wgpu::Origin3d::ZERO,
			aspect: wgpu::TextureAspect::All,
		},
		wgpu::TexelCopyBufferInfo {
			buffer: &buffer,
			layout: wgpu::TexelCopyBufferLayout {
				offset: 0,
				bytes_per_row: Some(padded_bytes_per_row),
				rows_per_image: Some(height),
			},
		},
		wgpu::Extent3d {
			width,
			height,
			depth_or_array_layers: 1,
		},
	);

	queue.submit(Some(encoder.finish()));

	// 同步等待: map_async + poll.
	//
	// **为什么用 mpsc channel 而非轮询 map 状态**:
	// wgpu 的 map_async 是回调式 API, 没有同步查询方法. 用 channel
	// 把回调结果转成阻塞等待, 是惯用做法.
	let slice = buffer.slice(..);
	let (tx, rx) = std::sync::mpsc::channel();
	slice.map_async(wgpu::MapMode::Read, move |result| {
		let _ = tx.send(result);
	});

	device
		.poll(wgpu::PollType::Wait {
			submission_index: None,
			timeout: None,
		})
		.map_err(|_| Error::ReadbackPollFailed)?;

	rx.recv()
		.map_err(|_| Error::ReadbackMapFailed)?
		.map_err(|_| Error::ReadbackMapFailed)?;

	// 去掉行 padding, 复制出紧密排列的像素数据.
	let mapped = slice
		.get_mapped_range()
		.map_err(|_| Error::ReadbackMapFailed)?;

	let mut pixels = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
	for row in 0..height {
		let start = (row * padded_bytes_per_row) as usize;
		let end = start + unpadded_bytes_per_row as usize;
		pixels.extend_from_slice(&mapped[start..end]);
	}
	drop(mapped);

	buffer.unmap();

	Ok(pixels)
}

/// 从 [`RenderTarget`] 读回颜色附件内容.
///
/// # 行为
///
/// 读取 `target.color_texture` —— 启用 MSAA 时它是解析后的结果, 未启用
/// MSAA 时它就是颜色纹理本身. 两种情况下的 `sample_count` 都是 1, 因此
/// 可以直接使用 `COPY_SRC`.
///
/// # 前置条件
///
/// `target` 由 [`RenderTarget::new`] 或 [`RenderTarget::new_msaa`] 创建,
/// 两者都声明了 `COPY_SRC` 用途.
pub fn read_target(
	device: &wgpu::Device,
	queue: &wgpu::Queue,
	target: &RenderTarget,
) -> Result<Vec<u8>> {
	read_texture(
		device,
		queue,
		&target.color_texture,
		target.width,
		target.height,
		target.color_format.wgpu(),
	)
}

/// 从缓冲读回字节数据.
///
/// # 行为
///
/// 创建一个 `MAP_READ` 的 staging buffer, 提交
/// `copy_buffer_to_buffer`, 等待完成后读出内容.
///
/// # 参数
///
/// - `buffer`: 源缓冲, 必须带 `COPY_SRC` 用途.
/// - `size`: 要读取的字节数, 从 offset 0 开始.
///
/// # 用途
///
/// - 读回 compute shader 的输出.
/// - 验证 vertex / index buffer 上传的正确性.
/// - 任何需要 CPU 端访问 GPU 缓冲的场景.
///
/// # 错误
///
/// - 缓冲映射失败: [`Error::ReadbackMapFailed`].
/// - 等待 GPU 完成失败: [`Error::ReadbackPollFailed`].
///
/// # 大小
///
/// `size` 必须是 4 的倍数 —— wgpu 对 `copy_buffer_to_buffer` 的大小有
/// 对齐要求, 由调用方保证.
pub fn read_buffer(
	device: &wgpu::Device,
	queue: &wgpu::Queue,
	buffer: &wgpu::Buffer,
	size: u64,
) -> Result<Vec<u8>> {
	if size == 0 {
		return Ok(Vec::new());
	}

	let staging = device.create_buffer(&wgpu::BufferDescriptor {
		label: Some("blaze3d_readback_staging"),
		size,
		usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
		mapped_at_creation: false,
	});

	let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
		label: Some("blaze3d_readback_encoder"),
	});

	encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);

	queue.submit(Some(encoder.finish()));

	let slice = staging.slice(..);
	let (tx, rx) = std::sync::mpsc::channel();
	slice.map_async(wgpu::MapMode::Read, move |result| {
		let _ = tx.send(result);
	});

	device
		.poll(wgpu::PollType::Wait {
			submission_index: None,
			timeout: None,
		})
		.map_err(|_| Error::ReadbackPollFailed)?;

	rx.recv()
		.map_err(|_| Error::ReadbackMapFailed)?
		.map_err(|_| Error::ReadbackMapFailed)?;

	let mapped = slice
		.get_mapped_range()
		.map_err(|_| Error::ReadbackMapFailed)?;

	let bytes = mapped.to_vec();
	drop(mapped);
	staging.unmap();

	Ok(bytes)
}

/// 计算未压缩格式的每像素字节数.
///
/// # 错误
///
/// - 格式是压缩格式 (块尺寸不是 1x1): [`Error::ReadbackUnsupportedFormat`].
/// - 格式无法拷贝到 buffer: 同上.
fn bytes_per_pixel(format: wgpu::TextureFormat) -> Result<u32> {
	// 压缩格式的块尺寸不是 1x1 —— `block_copy_size` 返回的是每块字节数,
	// 不是每像素字节数. 这里用块尺寸区分.
	//
	// wgpu 的 `block_dimensions()` 返回 (width, height): 未压缩格式是
	// (1, 1), BC / ETC / ASTC 是 (4, 4) 或更大.
	if format.block_dimensions() != (1, 1) {
		return Err(Error::ReadbackUnsupportedFormat { format });
	}

	format
		.block_copy_size(None)
		.ok_or(Error::ReadbackUnsupportedFormat { format })
}

/// 把 `n` 向上对齐到 `align` 的倍数.
fn align_to(n: u32, align: u32) -> u32 {
	n.div_ceil(align) * align
}
