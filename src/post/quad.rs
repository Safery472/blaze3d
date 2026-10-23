//! 全屏四边形.
//!
//! [`FullscreenQuad`] 是一个覆盖 NDC `[-1, 1]` 全屏范围的四边形网格,
//! UV 从 `(0, 0)` 到 `(1, 1)`, 与屏幕空间一一对应.
//!
//! # 顶点格式
//!
//! 使用 [`crate::vertex::formats::position_tex`], 即 Position + UV0.
//! 后处理着色器通常只需要这两个属性.
//!
//! # 与相机的配合
//!
//! 全屏四边形的顶点坐标已经是 NDC, **不需要**做 `view_projection`
//! 变换. 后处理着色器应直接把 position 输出到 `clip_position`.
//!
//! 因此 [`crate::shader::builtin`] 里有一个专门的后处理顶点着色器: 它
//! 读取 `Object.model` 但使用单位矩阵, 并且**不乘** `view_projection`.

use crate::state::ColorFormat;
use crate::system::GpuMesh;
use crate::vertex::{formats, BufferBuilder, DrawMode, VertexConsumer};

/// 全屏四边形网格.
///
/// # 复用
///
/// 所有后处理 pass 共享同一个四边形: 启动阶段创建一次, 之后只传递引用.
pub struct FullscreenQuad {
	mesh: GpuMesh,
}

impl FullscreenQuad {
	/// 创建全屏四边形.
	///
	/// # 顶点
	///
	/// 共 4 个顶点, 使用 [`crate::vertex::DrawMode::Quads`] 模式; 索引由
	/// 共享索引缓冲自动生成.
	///
	/// # UV 方向
	///
	/// - 顶点 `(-1, -1)` (NDC 左下) → UV `(0, 1)` (纹理左下).
	/// - 顶点 `(-1, 1)` (NDC 左上) → UV `(0, 0)` (纹理左上).
	///
	/// NDC 的 Y 向上, 纹理的 V 向下 —— 这个映射让纹理正确显示.
	pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, crate::system::Error> {
		let mut builder = BufferBuilder::with_capacity(4 * 20);
		let mut b = builder.begin(DrawMode::Quads, formats::position_tex());

		// 逆时针绕序 —— 与默认的 `CullMode::Back` 匹配. 后处理通常用
		// `CullMode::None`, 这里保持一致即可.
		//
		// 左下角.
		b.vertex(-1.0, -1.0, 0.0).expect("vertex");
		b.uv(0.0, 1.0).expect("uv");
		b.end_vertex().expect("end_vertex");

		// 右下角.
		b.vertex(1.0, -1.0, 0.0).expect("vertex");
		b.uv(1.0, 1.0).expect("uv");
		b.end_vertex().expect("end_vertex");

		// 右上角.
		b.vertex(1.0, 1.0, 0.0).expect("vertex");
		b.uv(1.0, 0.0).expect("uv");
		b.end_vertex().expect("end_vertex");

		// 左上角.
		b.vertex(-1.0, 1.0, 0.0).expect("vertex");
		b.uv(0.0, 0.0).expect("uv");
		b.end_vertex().expect("end_vertex");

		let rendered = b.end();
		let mesh = GpuMesh::new(device, queue, &rendered)?;

		Ok(Self { mesh })
	}

	/// 返回底层的网格引用.
	pub fn mesh(&self) -> &GpuMesh {
		&self.mesh
	}
}

/// 全屏四边形通常使用的 [`RenderType`].
///
/// 与 [`crate::vertex::formats::position_tex`] 匹配: 无深度, 不剔除.
/// 调用方需要覆盖 `color_format` 以匹配渲染目标.
///
/// [`RenderType`]: crate::state::RenderType
pub fn fullscreen_render_type(
	color_format: ColorFormat,
	name: &'static str,
) -> crate::state::RenderType {
	use crate::state::{CullMode, DepthMode, RenderType};

	RenderType::builder(name)
		.vertex_format(formats::position_tex())
		.color_format(color_format)
		.depth(DepthMode::disabled())
		.depth_format(None)
		.cull(CullMode::None)
		.build()
}
