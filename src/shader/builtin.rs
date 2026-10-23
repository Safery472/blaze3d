//! 开箱即用的内置着色器集合.
//!
//! 本模块给出一组**开箱即用**的 WGSL 着色器, 与 [`crate::vertex`] 的
//! 便利格式以及 [`crate::state::LayoutHandle`] 的 group 约定配套使用.
//!
//! # 与「不提供内置着色器」的关系
//!
//! 基础模块 [`super`] 刻意**不**内置任何着色器: 着色器与
//! `Globals` / `Object` 的布局强耦合, 属于应用层决策. 本模块在此之上
//! 提供**一套具体的, 有明确契约的**内置集合:
//!
//! - 契约写在每个着色器的文档里 (`bind group` 约定, 顶点格式).
//! - 覆盖范围只限最常见的组合, 不追求完备.
//! - 应用层可以完全不用本模块, 自行编写着色器.
//!
//! # ID 分配
//!
//! [`BuiltinShader::id`] 为每个内置着色器分配**固定**的 `id` (从
//! [`ID_BASE`] 起). 固定 `id` 的好处是: 多次创建同一个内置着色器只编译
//! 一次 —— 管线缓存按 `id` 命中. 应用层自定义着色器的 `id` 应避开这个
//! 区间.
//!
//! # 用法
//!
//! ```ignore
//! let shader = BuiltinShader::PositionColor.create(&device);
//! // 用 system 的 globals/object layout 建管线布局:
//! let layout = LayoutHandle::new(&device, &[
//!     system.globals_bind_group_layout(),
//!     system.object_bind_group_layout(),
//! ]);
//! ```
//!
//! # 着色器清单
//!
//! | 变体 | 顶点格式 | group 1 | group 2 | 用途 |
//! |---|---|---|---|---|
//! | [`BuiltinShader::PositionColor`] | `position_color` | `Object` | — | 纯顶点色 |
//! | [`BuiltinShader::PositionColorInstanced`] | `position_color` | instances | — | 实例化绘制 |
//! | [`BuiltinShader::PositionColorSkinned`] | `position_color_skinned` | instances | bones | 蒙皮 |
//! | [`BuiltinShader::PositionTex`] | `position_tex` | `Object` | texture | 纹理与雾 |
//! | [`BuiltinShader::PostCopy`] | `position_tex` | texture | — | 直通复制 (占位) |

use super::Shader;

/// 内置着色器 `id` 的起始值.
///
/// 应用层自定义着色器的 `id` 应**小于**这个值, 或明显大于区间上界,
/// 以免与内置的固定 `id` 冲突. 内置区间是
/// `[ID_BASE, ID_BASE + BuiltinShader::ALL.len())`.
pub const ID_BASE: u64 = 1 << 32;

/// 内置着色器枚举.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinShader {
	/// 纯顶点色: 顶点格式 `position_color`.
	///
	/// # Bind group
	///
	/// - group 0: `Globals`.
	/// - group 1: `Object` (`model`).
	PositionColor,

	/// 实例化顶点色: 顶点格式 `position_color`.
	///
	/// # Bind group
	///
	/// - group 0: `Globals`.
	/// - group 1: `instances` (`array<Instance>`, storage read).
	PositionColorInstanced,

	/// 蒙皮顶点色: 顶点格式 `position_color_skinned`.
	///
	/// # Bind group
	///
	/// - group 0: `Globals`.
	/// - group 1: `instances` (storage read).
	/// - group 2: `bones` (`array<mat4x4<f32>>`, storage read).
	PositionColorSkinned,

	/// 纹理与雾: 顶点格式 `position_tex`.
	///
	/// # Bind group
	///
	/// - group 0: `Globals`.
	/// - group 1: `Object` (`model`).
	/// - group 2: 基础颜色纹理 (binding 0 为 texture, 1 为 sampler).
	PositionTex,

	/// 后处理占位: 把输入原样复制到输出.
	///
	/// # Bind group
	///
	/// - group 0: `Globals` (布局保留, 未使用).
	/// - group 1: 输入纹理 (binding 0 为 texture, 1 为 sampler).
	///
	/// # 顶点格式
	///
	/// 顶点格式为 `position_tex`, 供后处理链 / 图使用.
	PostCopy,
}

impl BuiltinShader {
	/// 全部内置着色器构成的数组.
	pub const ALL: [BuiltinShader; 5] = [
		BuiltinShader::PositionColor,
		BuiltinShader::PositionColorInstanced,
		BuiltinShader::PositionColorSkinned,
		BuiltinShader::PositionTex,
		BuiltinShader::PostCopy,
	];

	/// 固定的管线缓存 `id`.
	///
	/// 同一个变体每次都返回相同值 —— 多次创建只编译一次管线.
	pub const fn id(self) -> u64 {
		match self {
			BuiltinShader::PositionColor => ID_BASE,
			BuiltinShader::PositionColorInstanced => ID_BASE + 1,
			BuiltinShader::PositionColorSkinned => ID_BASE + 2,
			BuiltinShader::PositionTex => ID_BASE + 3,
			BuiltinShader::PostCopy => ID_BASE + 4,
		}
	}

	/// 用于调试的标签.
	pub const fn label(self) -> &'static str {
		match self {
			BuiltinShader::PositionColor => "builtin_position_color",
			BuiltinShader::PositionColorInstanced => "builtin_position_color_instanced",
			BuiltinShader::PositionColorSkinned => "builtin_position_color_skinned",
			BuiltinShader::PositionTex => "builtin_position_tex",
			BuiltinShader::PostCopy => "builtin_post_copy",
		}
	}

	/// 该变体对应的 WGSL 源码.
	pub const fn source(self) -> &'static str {
		match self {
			BuiltinShader::PositionColor => {
				include_str!("../../shaders/position_color.min.wgsl")
			}
			BuiltinShader::PositionColorInstanced => {
				include_str!("../../shaders/position_color_instanced.min.wgsl")
			}
			BuiltinShader::PositionColorSkinned => {
				include_str!("../../shaders/position_color_skinned.min.wgsl")
			}
			BuiltinShader::PositionTex => {
				include_str!("../../shaders/position_tex.min.wgsl")
			}
			BuiltinShader::PostCopy => {
				include_str!("../../shaders/post_copy.min.wgsl")
			}
		}
	}

	/// 创建对应变体的着色器.
	///
	/// # 行为
	///
	/// 用固定 `id` 创建 [`Shader`] —— 见 [`Self::id`].
	pub fn create(self, device: &wgpu::Device) -> Shader {
		Shader::from_wgsl(device, self.id(), self.source(), Some(self.label()))
	}
}

// ================ 内置布局约定 ================

/// 内置着色器期望的 `bind group` 约定说明.
///
/// 提供文档与校验辅助 —— 帮助调用方构造出匹配的
/// [`crate::state::LayoutHandle`].
impl BuiltinShader {
	/// 该着色器需要的 `bind group` 个数 (包含 group 0).
	///
	/// 调用方构造管线布局时至少要给出这么多个 group:
	///
	/// - group 0: 恒为 `Globals`.
	/// - group 1: `Object` 或实例 / 纹理 (见变体文档).
	/// - group 2: 额外的 `instances` / `bones` / 纹理.
	pub const fn bind_group_count(self) -> u32 {
		match self {
			BuiltinShader::PositionColor => 2,
			BuiltinShader::PositionColorInstanced => 2,
			BuiltinShader::PositionColorSkinned => 3,
			BuiltinShader::PositionTex => 3,
			BuiltinShader::PostCopy => 2,
		}
	}

	/// 顶点格式用到的 `shader_location` 个数.
	///
	/// 用于检查管线布局里声明的顶点属性数是否对得上.
	pub const fn vertex_attribute_count(self) -> u32 {
		match self {
			// position 与 color.
			BuiltinShader::PositionColor
			| BuiltinShader::PositionColorInstanced => 2,
			// position, color, bone_index, bone_weight.
			BuiltinShader::PositionColorSkinned => 4,
			// position 与 uv.
			BuiltinShader::PositionTex | BuiltinShader::PostCopy => 2,
		}
	}
}
