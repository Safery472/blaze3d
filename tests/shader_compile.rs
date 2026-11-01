// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 着色器编译回归测试.
//!
//! 所有 `shaders/*.min.wgsl` 都通过 `include_str!` 内嵌进库, 语法错误
//! 只会在运行到对应路径时才暴露 —— 而那条路径往往需要特定格式或
//! 特定绘制方式才能走到. 本测试在真实设备上把所有内嵌着色器编译一遍,
//! 把这类错误提前到 `cargo test`.
//!
//! # 设备
//!
//! 优先用软件后端 (`llvmpipe` / `lavapipe` / WARP), 拿不到时退回任一
//! 可用设备. **拿不到任何设备时整个测试静默通过** —— 无 GPU 的 CI
//! 不应因此失败, 有 GPU 的机器上则一定会跑到.

// wgpu 的类型图较深, 证明 `Send` 时会超出默认的 trait 求解深度.
#![recursion_limit = "256"]

use std::collections::HashSet;

/// 需要编译的内嵌着色器: (名称, 源码).
///
/// 新增 `include_str!` 时把对应常量加进来 —— 这是唯一一处需要手工
/// 同步的地方.
fn embedded_shaders() -> Vec<(&'static str, String)> {
	let literal = |source: &'static str| source.to_string();
	vec![
		(
			"position_color",
			literal(include_str!("../shaders/position_color.min.wgsl")),
		),
		(
			"position_color_instanced",
			literal(include_str!("../shaders/position_color_instanced.min.wgsl")),
		),
		(
			"position_color_skinned",
			literal(include_str!("../shaders/position_color_skinned.min.wgsl")),
		),
		("position_tex", literal(include_str!("../shaders/position_tex.min.wgsl"))),
		("post_copy", literal(include_str!("../shaders/post_copy.min.wgsl"))),
		("post_vignette", literal(include_str!("../shaders/post_vignette.min.wgsl"))),
		("double", literal(include_str!("../shaders/double.min.wgsl"))),
		("skybox", literal(include_str!("../shaders/skybox.min.wgsl"))),
		(
			"shadow_depth",
			literal(include_str!("../shaders/shadow_depth.min.wgsl")),
		),
		(
			"shadow_depth_mesh",
			literal(include_str!("../shaders/shadow_depth_mesh.min.wgsl")),
		),
		(
			"shadow_depth_skinned",
			literal(include_str!("../shaders/shadow_depth_skinned.min.wgsl")),
		),
		(
			"mipmap",
			literal(include_str!("../shaders/mipmap.min.wgsl"))
				.replace("{STORAGE_FORMAT}", "rgba8unorm"),
		),
		(
			"mipmap_3d",
			literal(include_str!("../shaders/mipmap_3d.min.wgsl"))
				.replace("{STORAGE_FORMAT}", "rgba8unorm"),
		),
		(
			"mipmap_array",
			literal(include_str!("../shaders/mipmap_array.min.wgsl"))
				.replace("{STORAGE_FORMAT}", "rgba8unorm")
				.replace("{VTYPE}", "f32"),
		),
		(
			"mipmap_integer",
			literal(include_str!("../shaders/mipmap_integer.min.wgsl"))
				.replace("{STORAGE_FORMAT}", "rgba8uint")
				.replace("{VTYPE}", "u32"),
		),
		(
			"mipmap_integer_3d",
			literal(include_str!("../shaders/mipmap_integer_3d.min.wgsl"))
				.replace("{STORAGE_FORMAT}", "rgba8uint")
				.replace("{VTYPE}", "u32"),
		),
		(
			"mipmap_array_integer",
			literal(include_str!("../shaders/mipmap_array_integer.min.wgsl"))
				.replace("{STORAGE_FORMAT}", "rgba8uint")
				.replace("{VTYPE}", "u32"),
		),
	]
}

/// 取一个可用设备; 没有 GPU 时返回 `None`.
fn device() -> Option<wgpu::Device> {
	let instance =
		wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

	let request = |fallback: bool| {
		pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
			power_preference: wgpu::PowerPreference::LowPower,
			compatible_surface: None,
			force_fallback_adapter: fallback,
			apply_limit_buckets: false,
		}))
		.ok()
	};

	let adapter = request(true).or_else(|| request(false))?;
	let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
		label: Some("shader_tests"),
		required_features: wgpu::Features::empty(),
		required_limits: wgpu::Limits::default(),
		..Default::default()
	}))
	.ok()?;
	Some(device)
}

/// 统计源码里出现的入口点.
///
/// 只做最粗粒度的检查: 每个着色器至少要有一个 `@vertex` 或
/// `@fragment` 或 `@compute`. 真正的语法检查交给 wgpu 编译.
fn entry_point_count(source: &str) -> usize {
	source.matches("@vertex").count()
		+ source.matches("@fragment").count()
		+ source.matches("@compute").count()
}

#[test]
fn embedded_shaders_compile() {
	let Some(device) = device() else {
		eprintln!("no GPU adapter available; skipping embedded shader compilation");
		return;
	};

	let mut failures = Vec::new();

	for (name, source) in embedded_shaders() {
		assert!(
			entry_point_count(&source) > 0,
			"{name}: no @vertex/@fragment/@compute entry point found"
		);

		// `push_error_scope` 把编译错误转成可读取的结果, 而不是打日志
		// 之后被丢弃.
		let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
		let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some(name),
			source: wgpu::ShaderSource::Wgsl(source.into()),
		});
		drop(module);

		if let Some(error) = pollster::block_on(scope.pop()) {
			failures.push(format!("{name}: {error}"));
		}
	}

	assert!(failures.is_empty(), "shader compilation failed:\n{}", failures.join("\n"));
}

/// 可读版与压缩版必须存在同名文件 —— `Style.md` 要求两者同步维护.
#[test]
fn every_shader_has_a_readable_and_a_minified_copy() {
	let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");
	let mut names = HashSet::new();

	for entry in std::fs::read_dir(dir).expect("read shaders dir") {
		let path = entry.expect("dir entry").path();
		let file = path.file_name().and_then(|n| n.to_str()).expect("utf-8 name").to_string();
		if let Some(stem) = file.strip_suffix(".min.wgsl") {
			names.insert((stem.to_string(), "min"));
		} else if let Some(stem) = file.strip_suffix(".wgsl") {
			names.insert((stem.to_string(), "readable"));
		}
	}

	let readable: HashSet<_> =
		names.iter().filter(|(_, kind)| *kind == "readable").map(|(n, _)| n).collect();
	let minified: HashSet<_> =
		names.iter().filter(|(_, kind)| *kind == "min").map(|(n, _)| n).collect();

	assert_eq!(
		readable, minified,
		"every shaders/*.wgsl must have a matching *.min.wgsl"
	);
}

/// 压缩版必须由 `tools/wgsl-min` 生成, 不能是手写的.
///
/// `tools/wgsl-min --check` 是权威检查, 但它跑在另一个 workspace 里,
/// 普通的 `cargo test` 不会调用它 (而且需要外部 `nagami`). 这里复刻
/// 它的产物特征, 让手改立刻失败:
///
/// - **单行** —— nagami 的输出是单行最小文本.
/// - **没有注释** —— 手写版本几乎总会带上注释.
/// - **没有缩进或空行**.
///
/// 判据故意取得宽松: 目的是拦住手改, 而不去复刻 nagami 的逐字节
/// 输出 (那需要同一个二进制). 真正的等价性由工具自己的 `--check`
/// 保证.
#[test]
fn minified_shaders_are_generated_not_hand_written() {
	let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");
	let mut problems = Vec::new();

	for entry in std::fs::read_dir(dir).expect("read shaders dir") {
		let path = entry.expect("dir entry").path();
		let Some(file) = path.file_name().and_then(|n| n.to_str()) else {
			continue;
		};
		if !file.ends_with(".min.wgsl") {
			continue;
		}

		let source = std::fs::read_to_string(&path).expect("read minified shader");

		if source.lines().count() != 1 {
			problems.push(format!(
				"{file}: spans {} lines; nagami emits a single line",
				source.lines().count()
			));
		}
		if source.contains("//") || source.contains("/*") {
			problems.push(format!("{file}: contains a comment"));
		}
		if source.lines().any(|line| line.trim_start() != line || line.trim().is_empty()) {
			problems.push(format!("{file}: contains indentation or a blank line"));
		}
	}

	assert!(
		problems.is_empty(),
		"these files look hand-written; regenerate them with\n  \
		 cargo run --manifest-path tools/wgsl-min/Cargo.toml\n{}",
		problems.join("\n")
	);
}
