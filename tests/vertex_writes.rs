// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 顶点写入路径的 CPU 回归测试.
//!
//! 顶点格式占 4 字节槽位这件事, 只有走到"写入后再读回字节"才发现得了
//! 错位; 这里不碰 GPU, 直接检查 `BufferBuilder` 产出的原始字节.

use blaze3d::vertex::{
	BufferBuilder, DrawMode, VertexConsumer, VertexFormat, VertexFormatElement,
};

/// 只含法线的格式: 每个顶点 4 字节, 前 3 字节是 i8 法线, 第 4 字节补 0.
fn normal_only() -> VertexFormat {
	VertexFormat::new([("Normal", VertexFormatElement::normal_i8x3())]).expect("format")
}

#[test]
fn normal_writes_all_four_bytes_of_its_slot() {
	let mut builder = BufferBuilder::with_capacity(64);
	let mut b = builder.begin(DrawMode::Triangles, normal_only());

	b.normal(1.0, 0.0, 0.0).expect("normal");
	b.end_vertex().expect("end vertex");
	b.normal(0.0, -1.0, 0.0).expect("normal");
	b.end_vertex().expect("end vertex");

	let rendered = b.end();

	// i8 的 1.0 映射为 127, -1.0 映射为 -127 (0x81), 0.0 为 0.
	assert_eq!(
		rendered.vertex_data(),
		&[127, 0, 0, 0, 0, 0x81, 0, 0],
		"每个法线都必须占满 4 字节槽位, 第 4 个分量为 0"
	);
}

#[test]
fn normal_after_position_keeps_offsets_aligned() {
	let format = VertexFormat::new([
		("Position", VertexFormatElement::position_f32x3()),
		("Normal", VertexFormatElement::normal_i8x3()),
	])
	.expect("format");
	assert_eq!(format.vertex_size(), 16);

	let mut builder = BufferBuilder::with_capacity(64);
	let mut b = builder.begin(DrawMode::Triangles, format);

	b.vertex(1.0, 2.0, 3.0).expect("vertex");
	b.normal(0.0, 1.0, 0.0).expect("normal");
	b.end_vertex().expect("end vertex");
	b.vertex(4.0, 5.0, 6.0).expect("vertex");
	b.normal(0.0, 0.0, -1.0).expect("normal");
	b.end_vertex().expect("end vertex");

	let rendered = b.end();
	let data = rendered.vertex_data();

	assert_eq!(data.len(), 32);
	// 第二个顶点的位置从第 16 字节开始 —— 法线少写一个字节就会错位.
	assert_eq!(data[16..20], 4.0_f32.to_le_bytes());
	// 第二个顶点的法线在顶点内偏移 12, 即第 28..32 字节.
	assert_eq!(data[28..32], [0, 0, 0x81, 0]);
}
