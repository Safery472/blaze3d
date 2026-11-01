// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 由可读版 `shaders/*.wgsl` 生成压缩版 `shaders/*.min.wgsl`.
//!
//! # 为什么用 nagami
//!
//! 早先这里自带一套"剥注释 + 折叠空白"的压缩器. 那只是**文本瘦身**:
//! `.min.wgsl` 仍是逐行可读的源码, 标识符一个没短, 死代码一个没删.
//! 现在改用 [`nagami`](https://github.com/kriss-spy/nagami): 它把 WGSL
//! 降到 naga IR, 跑一整套优化 pass (死代码消除, 常量折叠, 复制传播,
//! 内联等), 再以单行紧凑形式吐出最小合法 WGSL.
//!
//! # 参数
//!
//! - `--profile max`: 开启最大优化. 体积优先.
//! - `--no-mangle` **必须加**. 默认会改写 struct 成员名, 而库的
//!   Rust 侧按 `view_projection` / `model` 之类的**名字**计算 uniform
//!   布局并对照文档; 改成 `a` / `b` 会让文档与代码对不上. 函数参数与
//!   局部变量仍然会被缩成单字母, 所以收益的大头保留.
//! - `--strict-fallback` **必须加**. 默认遇到 naga 无法优化的输入会
//!   原样吐出**未优化**的源码并且退出码仍为 0 —— 那会静默写出一个
//!   比可读版还大的"压缩版". 加上后这类输入直接失败.
//!
//! 入口点名字 (`vs_main` / `fs_main` / `cs_main`) 会被保留, 库按名字取
//! `entry_point`. bind group 按 binding 序号匹配, 不依赖变量名.
//!
//! # 占位符
//!
//! 模板里的 `{STORAGE_FORMAT}` / `{VTYPE}` 不是合法 WGSL, nagami 无法
//! 解析. 生成时按表格替换成一组确定的真实值, 压缩完**再把占位符换回去**.
//!
//! 换回是安全的: 占位符只出现在类型位置 (`texture_storage_2d<{STORAGE_FORMAT}, write>`,
//! `texture_2d<{VTYPE}>`, `{VTYPE}(4)`), 这些位置都是标量或格式名,
//! nagami 不会改写它们, 替换前后的一一对应关系因此保持.
//!
//! # 用法
//!
//! ```bash
//! # 重新生成全部 *.min.wgsl
//! cargo run --manifest-path tools/wgsl-min/Cargo.toml
//!
//! # 只校验压缩版是否落后于可读版 (适合放 CI)
//! cargo run --manifest-path tools/wgsl-min/Cargo.toml -- --check
//! ```
//!
//! 依赖外部 `nagami` 可执行文件, 默认从 `PATH` 找, 也可以用
//! `NAGAMI=/path/to/nagami` 覆盖. 本工具是独立 workspace, 不进入
//! blaze3d 的依赖图.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// `{VTYPE}` 的替换值, 按"先无符号后有符号"排列.
///
/// 整数格式分成 `uint` / `sint` 两类, 而 nagami 会把抽象类型固化成
/// 具体类型 —— 两类**都必须试**, 否则写出的模板在另一类格式上编译
/// 不过. 两条探针的产物还原后必须一致 (见 [`minify`]).
const VTYPE_PROBES: [&str; 2] = ["u32", "i32"];

/// `{STORAGE_FORMAT}` 的替换值, 与 [`VTYPE_PROBES`] 一一对应.
const STORAGE_PROBES: [&str; 2] = ["rgba8uint", "rgba8sint"];

/// 只含 `{STORAGE_FORMAT}`、没有 `{VTYPE}` 的模板使用的浮点 storage 格式.
///
/// 这类模板 (`mipmap.wgsl` / `mipmap_3d.wgsl`) 用 `texture_2d<f32>` 与
/// `textureSampleLevel`, 只能配浮点 storage 格式.
const FLOAT_STORAGE_PROBE: &str = "rgba32float";

fn main() -> ExitCode {
	let args: Vec<String> = std::env::args().skip(1).collect();
	let check = args.iter().any(|a| a == "--check");

	let nagami = std::env::var("NAGAMI").unwrap_or_else(|_| "nagami".to_string());
	if Command::new(&nagami)
		.arg("--version")
		.output()
		.map(|o| !o.status.success())
		.unwrap_or(true)
	{
		eprintln!(
			"error: cannot run `{nagami}`; install it with `cargo install nagami` \
			 or point NAGAMI at the binary"
		);
		return ExitCode::FAILURE;
	}

	let Some(dir) = shader_dir() else {
		eprintln!("error: cannot locate the shaders directory");
		return ExitCode::FAILURE;
	};

	let mut sources = match collect_sources(&dir) {
		Ok(sources) => sources,
		Err(e) => {
			eprintln!("error: cannot read {}: {e}", dir.display());
			return ExitCode::FAILURE;
		}
	};
	sources.sort();

	let mut errors = Vec::new();
	let mut written = 0usize;

	for source in sources {
		let Some(stem) = source.file_stem().and_then(|s| s.to_str()).map(str::to_string) else {
			errors.push(format!("{}: not a UTF-8 file name", source.display()));
			continue;
		};

		let readable = match fs::read_to_string(&source) {
			Ok(text) => text,
			Err(e) => {
				errors.push(format!("{stem}.wgsl: {e}"));
				continue;
			}
		};

		let minified = match minify(&nagami, &readable) {
			Ok(text) => text,
			Err(e) => {
				errors.push(format!("{stem}.wgsl: {e}"));
				continue;
			}
		};

		let out_path = dir.join(format!("{stem}.min.wgsl"));

		if check {
			let existing = fs::read_to_string(&out_path).unwrap_or_default();
			if existing != minified {
				errors.push(format!("{stem}.min.wgsl is out of date; rerun without --check"));
			}
		} else if let Err(e) = fs::write(&out_path, &minified) {
			errors.push(format!("{}: {e}", out_path.display()));
		} else {
			written += 1;
			println!(
				"{stem}.wgsl -> {stem}.min.wgsl ({} -> {} bytes)",
				readable.len(),
				minified.len()
			);
		}
	}

	if errors.is_empty() {
		if !check {
			println!("wrote {written} file(s)");
		}
		ExitCode::SUCCESS
	} else {
		for error in errors {
			eprintln!("error: {error}");
		}
		ExitCode::FAILURE
	}
}

/// 仓库根的 `shaders/` 目录.
fn shader_dir() -> Option<PathBuf> {
	PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.parent()
		.and_then(Path::parent)
		.map(|root| root.join("shaders"))
}

/// 列出所有可读版着色器 (`*.wgsl`, 排除 `*.min.wgsl`).
fn collect_sources(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
	let mut sources = Vec::new();
	for entry in fs::read_dir(dir)? {
		let path = entry?.path();
		let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
		if name.ends_with(".wgsl") && !name.ends_with(".min.wgsl") {
			sources.push(path);
		}
	}
	Ok(sources)
}

/// 把一段模板源码压成最小版.
///
/// # 为什么两种 `{VTYPE}` 都要试
///
/// nagami 会把 WGSL 的抽象类型**固化到具体类型**: 模板里的
/// `vec4<{VTYPE}>` 会被它根据探针写成 `vec4u` 或 `vec4i`. 只试一种
/// 探针, 写出的 `.min.wgsl` 就只能在那一类格式上编译 —— 整数格式分成
/// `uint` 与 `sint` 两类, 两者都会走到这些模板.
///
/// 因此这里对每个探针各跑一次, 都还原占位符后要求**两者完全一致**;
/// 只要有一份成功就把那份写出去, 并显式比对两种探针的产物:
///
/// - 一致: 说明还原之后模板对两类格式都成立, 可以共享一个文件.
/// - 不一致: 说明 nagami 的优化结果依赖具体类型, 无法用同一份文本
///   服务两类格式 —— 报错而不是挑一份写出去.
///
/// # 还原顺序
///
/// 先还原向量类型 (`vec4u` / `vec4i` / `vec4f` -> `vec4<{VTYPE}>`),
/// 再还原标量与格式名. 前者是 nagami 的别名缩写, 不还原则模板对另一类
/// 整数格式就是错的.
fn minify(nagami: &str, source: &str) -> Result<String, String> {
	let needs_vtype = source.contains("{VTYPE}");
	let needs_storage = source.contains("{STORAGE_FORMAT}");

	if !needs_vtype && !needs_storage {
		return run_nagami(nagami, source);
	}

	// 没有 `{VTYPE}` 的模板只服务浮点格式 (它们用 `texture_2d<f32>` 与
	// `textureSampleLevel`), 用整数 storage 格式去探只会让 nagami 在
	// 类型检查上失败. 这类模板只试一个浮点探针.
	let probes: &[usize] = if needs_vtype { &[0, 1] } else { &[0] };

	let mut outputs = Vec::new();
	let mut last_error = None;

	for &index in probes {
		let storage = if needs_vtype {
			STORAGE_PROBES[index]
		} else {
			FLOAT_STORAGE_PROBE
		};
		let mut probe = source.replace("{STORAGE_FORMAT}", storage);
		if needs_vtype {
			probe = probe.replace("{VTYPE}", VTYPE_PROBES[index]);
		}

		match run_nagami(nagami, &probe) {
			Ok(minified) => {
				let restored = if needs_vtype {
					restore(&minified, index, true, needs_storage)
				} else {
					minified.replace(FLOAT_STORAGE_PROBE, "{STORAGE_FORMAT}")
				};
				outputs.push(restored);
			}
			Err(e) => last_error = Some(e),
		}
	}

	match outputs.as_slice() {
		[] => Err(last_error.unwrap_or_else(|| "no probe succeeded".to_string())),
		[one] => Ok(one.clone()),
		[first, rest @ ..] => {
			if rest.iter().all(|other| other == first) {
				Ok(first.clone())
			} else {
				Err(format!(
					"the {} probes disagree after restoring placeholders: nagami's \
					 output depends on the concrete type, so the template cannot be \
					 minified once for all of {}",
					VTYPE_PROBES.len(),
					VTYPE_PROBES.join(" / ")
				))
			}
		}
	}
}

/// 把探针的具体类型换回占位符.
///
/// `index` 决定用的是哪一组替换值; `needs_*` 决定该模板**真正**含有的
/// 占位符. 不做无条件替换: `mipmap.wgsl` 里有字面 `f32`, 若照样换回
/// `{VTYPE}`, 库运行时会再替换一次, 变成 `texture_2d<u32>` 与浮点
/// 采样器不匹配.
fn restore(minified: &str, index: usize, needs_vtype: bool, needs_storage: bool) -> String {
	let mut out = minified.to_string();

	if !needs_vtype && !needs_storage {
		return out;
	}

	if needs_vtype {
		// nagami 把 `vec4<{VTYPE}>` 缩成 `vec4u` / `vec4i` / `vec4f`.
		//
		// **只**还原 `vec4*`. 模板里其它向量 (`vec2<i32>` / `vec3<u32>`
		// 之类) 是坐标与维度的索引类型, nagami 同样会缩成 `vec2i` /
		// `vec3u`, 但它们与 `{VTYPE}` 无关 —— 把它们也换成
		// `vec2<{VTYPE}>` 会在 `u32` 格式下把层索引、坐标类型改错,
		// naga 报 "Composing 0's component type is not expected".
		for concrete in ['f', 'u', 'i'] {
			out = out.replace(&format!("vec4{concrete}"), "vec4<{VTYPE}>");
		}
	}
	if needs_storage {
		out = out.replace(STORAGE_PROBES[index], "{STORAGE_FORMAT}");
	}
	if needs_vtype {
		// **只**还原类型位置 `<u32>` / `<i32>`, 不还原 `i32(x)`.
		//
		// `mipmap_array*.wgsl` 里 `layer = i32(id.z)` 的 `i32` 是数组层
		// 索引的转换, 与采样类型无关. 若把它一并换成 `{VTYPE}`, 模板在
		// `u32` 格式下会变成 `u32(id.z)`, 层索引就错了.
		out = out.replace(&format!("<{}>", VTYPE_PROBES[index]), "<{VTYPE}>");
	}

	out
}

/// 调一次 nagami, 返回单行最小版.
fn run_nagami(nagami: &str, source: &str) -> Result<String, String> {
	let input = std::env::temp_dir().join(format!("wgsl-min-{}.wgsl", std::process::id()));
	fs::write(&input, source).map_err(|e| format!("cannot write {}: {e}", input.display()))?;

	let output = Command::new(nagami)
		.arg(&input)
		// stdout 即最终产物, 不落盘.
		.args(["-o", "-", "--profile", "max", "--no-mangle", "--strict-fallback"])
		.output();

	let _ = fs::remove_file(&input);

	let output = output.map_err(|e| format!("cannot run nagami: {e}"))?;

	if !output.status.success() {
		let stderr = String::from_utf8_lossy(&output.stderr);
		return Err(format!("nagami failed: {}", stderr.trim()));
	}

	let text = String::from_utf8(output.stdout).map_err(|e| format!("nagami output: {e}"))?;
	let text = text.trim_end();
	if text.is_empty() {
		return Err("nagami produced no output".to_string());
	}

	// nagami 正常输出本来就是单行; 这里再兜一次, 免得将来版本变了悄悄
	// 把多行写进 `.min.wgsl`.
	Ok(text.lines().map(str::trim).collect::<Vec<_>>().join(""))
}

#[cfg(test)]
mod tests {
	use super::{STORAGE_PROBES, VTYPE_PROBES, restore};

	/// 两种占位符都出现在**类型位置**时, 替换后能原样还原.
	#[test]
	fn both_placeholders_round_trip() {
		let source = "texture_2d<{VTYPE}> / vec4<{VTYPE}> / <{STORAGE_FORMAT}, write>";

		for index in 0..VTYPE_PROBES.len() {
			let probe = source
				.replace("{STORAGE_FORMAT}", STORAGE_PROBES[index])
				.replace("{VTYPE}", VTYPE_PROBES[index]);
			assert!(!probe.contains('{'), "probe still has a placeholder: {probe}");

			assert_eq!(restore(&probe, index, true, true), source);
		}
	}

	/// 只有 `{STORAGE_FORMAT}` 的模板 (例如 `mipmap.wgsl`): 里面的
	/// **字面** `f32` 必须原样保留, 不能被还原成 `{VTYPE}`.
	///
	/// 回归: 早先无条件替换两种替换值, 结果 `texture_2d<f32>` 变成
	/// `texture_2d<{VTYPE}>`, 库运行时会把它再替换成 `u32`, 与浮点
	/// 采样器不匹配, 着色器编译直接失败.
	#[test]
	fn storage_only_template_keeps_literal_f32() {
		let probe = "var t: texture_2d<f32>; var o: texture_storage_2d<rgba8uint, write>;";

		let restored = restore(probe, 0, false, true);

		assert!(
			restored.contains("texture_2d<f32>"),
			"literal f32 was rewritten: {restored}"
		);
		assert!(restored.contains("<{STORAGE_FORMAT}, write>"), "{restored}");
		assert!(!restored.contains("{VTYPE}"), "invented a {{VTYPE}}: {restored}");
	}

	/// nagami 把 `vec4<{VTYPE}>` 缩成 `vec4u` / `vec4i` / `vec4f`,
	/// 还原时必须展开回 `vec4<{VTYPE}>`, 否则写出的模板对另一类整数
	/// 格式是错的.
	///
	/// **同时**验证其它向量不被误还原: `vec2i` / `vec3u` 是坐标与层
	/// 索引的类型, 与 `{VTYPE}` 无关.
	#[test]
	fn concrete_vector_types_are_restored() {
		for index in 0..VTYPE_PROBES.len() {
			let scalar = VTYPE_PROBES[index];
			let probe =
				format!("var a: vec4u; var b: vec2i; var c: vec3u; var d: texture_2d<{scalar}>;");

			let restored = restore(&probe, index, true, false);

			assert!(restored.contains("vec4<{VTYPE}>"), "{restored}");
			assert!(!restored.contains("vec4u"), "{restored}");
			assert!(restored.contains("texture_2d<{VTYPE}>"), "{restored}");

			// 索引类型必须原样保留.
			assert!(restored.contains("var b: vec2i;"), "vec2i was rewritten: {restored}");
			assert!(restored.contains("var c: vec3u;"), "vec3u was rewritten: {restored}");
		}
	}

	/// 只有 `{VTYPE}` 的模板: 字面 `rgba8unorm` 同样不能被还原.
	#[test]
	fn vtype_only_template_keeps_literal_storage_name() {
		let probe = "var a: vec4u; var b: texture_storage_2d<rgba8unorm, write>;";

		let restored = restore(probe, 0, true, false);

		assert!(restored.contains("vec4<{VTYPE}>"), "{restored}");
		assert!(
			restored.contains("rgba8unorm, write"),
			"literal storage format was rewritten: {restored}"
		);
	}

	/// 没有占位符的模板完全不还原.
	#[test]
	fn plain_shader_is_untouched() {
		let probe = "let x = f32(4);";
		assert_eq!(restore(probe, 0, false, false), probe);
	}
}
