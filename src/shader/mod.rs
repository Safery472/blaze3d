// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! WGSL 着色器模块的封装.
//!
//! [`Shader`] 是 wgpu 着色器模块的包装, 同时记录一个用于管线缓存的 `id`.
//!
//! # `Shader` 的 `id`
//!
//! [`Shader::from_wgsl`] 接受调用方指定的 `id`. 该 `id` 参与
//! [`crate::state::PipelineDesc`] 的构造 —— 同一份 WGSL 源码用不同 `id`
//! 构造时, 会生成不同的管线缓存条目.
//!
//! 调用方需自行保证 `id` 唯一, 通常用一个递增计数器:
//!
//! ```ignore
//! use std::sync::atomic::{AtomicU64, Ordering};
//!
//! static NEXT_ID: AtomicU64 = AtomicU64::new(1);
//!
//! let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
//! let shader = Shader::from_wgsl(&device, id, source, Some("my_shader"));
//! ```
//!
//! # 不提供内置着色器 (基础模块)
//!
//! 本模块**不**提供任何内置 WGSL 源码. 原因如下:
//!
//! - 着色器与 `Globals` / `Object` 的布局强耦合, 属于应用层的决策.
//! - 各应用需要的着色器组合不同, 内置集合既臃肿又限制自由.
//! - 示例 / 测试所用的着色器应当跟着示例 / 测试走.
//!
//! 需要开箱即用的着色器时, 改用 [`builtin`] 子模块 —— 它在上面的原则
//! 之上提供**一套具体的, 有明确契约的**常用着色器, 应用层可以选择不用.

pub mod builtin;

pub use builtin::{BuiltinShader, ID_BASE as BUILTIN_ID_BASE};

use std::sync::Arc;

/// 着色器模块与它的管线缓存 `id`.
///
/// # 字段
///
/// - `id` —— 用于管线缓存的唯一标识.
/// - `module` —— wgpu 的着色器模块.
///
/// # 共享
///
/// `module` 用 `Arc` 包装 —— 同一个着色器可以在多个管线之间共享, 无需
/// 重复编译.
pub struct Shader {
	/// 管线缓存所用的唯一 `id`.
	pub id: u64,
	/// wgpu 的着色器模块.
	pub module: Arc<wgpu::ShaderModule>,
}

impl Shader {
	/// 用 WGSL 源码创建着色器模块.
	///
	/// # 参数
	///
	/// - **`device`**: 用于创建着色器模块的设备.
	/// - **`id`**: 管线缓存用的 `id`. 调用方负责保证同一个 `id` 对应同一份源码.
	/// - **`source`**: WGSL 源码.
	/// - **`label`**: 调试标签.
	///
	/// # 编译错误
	///
	/// wgpu 的着色器编译错误是异步上报的 —— `create_shader_module` 不会
	/// 当场 panic. 编译错误会在首次用该模块创建管线时暴露.
	///
	/// 若希望在创建阶段就捕获错误, 可以用
	/// [`wgpu::Device::push_error_scope`] / `pop_error_scope` 包裹本调用.
	pub fn from_wgsl(
		device: &wgpu::Device,
		id: u64,
		source: &str,
		label: Option<&str>,
	) -> Self {
		let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label,
			source: wgpu::ShaderSource::Wgsl(source.into()),
		});
		Self {
			id,
			module: Arc::new(module),
		}
	}
}
