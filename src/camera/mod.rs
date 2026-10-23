//! 相机抽象层.
//!
//! [`Camera`] trait 把相机状态统一成视图矩阵 / 投影矩阵查询接口, 由两个
//! 实现覆盖常见场景:
//!
//! - [`PerspectiveCamera`] —— 透视投影, 用于 3D 场景, 呈现近大远小.
//! - [`OrthographicCamera`] —— 正交投影, 用于 2D 场景或工程图, 远近同大.
//!
//! # 坐标系约定
//!
//! - **右手系**: 相机朝向 `-Z`, 与 wgpu / `glam` 的默认约定一致.
//! - **Y 轴向上**: 世界空间中的 Y 轴指向正上方.
//! - **NDC**: 深度范围为 `[0, 1]`, 与 wgpu / Vulkan 一致, 而非
//!   `OpenGL` 那样的 `[-1, 1]`.
//!
//! # 与 `Globals` 的关系
//!
//! [`Camera::apply`] 把相机的全部输出写入 [`crate::system::Globals`]:
//! 视图投影矩阵, 视图矩阵, 逆矩阵与相机位置. 调用方在渲染前先写入:
//!
//! ```ignore
//! camera.apply(render_system.globals_mut());
//! render_system.render_frame(&target, |pass| { ... });
//! ```
//!
//! # 视口变化
//!
//! 窗口尺寸变化后, 相机的 `aspect` 必须同步更新; 一旦滞后, 投影矩阵
//! 就会被拉伸. **`set_viewport` 不属于 [`Camera`] trait** —— 它由
//! [`PerspectiveCamera`] 与 [`OrthographicCamera`] 各自定义, 因为
//! 宽高比的存储方式属于实现细节 (`PerspectiveCamera` 与
//! `OrthographicCamera` 都直接持有 `aspect` 字段, trait 层访问不到).
//!
//! 调用方持有具体类型时, 直接调用即可:
//!
//! ```ignore
//! perspective_camera.set_viewport(window_width, window_height);
//! // 或
//! orthographic_camera.set_viewport(window_width, window_height);
//! ```
//!
//! 在泛型上下文 (`fn f<C: Camera>(camera: &mut C)`) 中更新视口时, 调用方
//! 需要自行定义扩展 trait, 或改用具体类型作为类型参数. 这是有意的设计
//! 取舍: [`Camera`] 保持**只读查询**语义, 不为视口更新这一个场景引入
//! `&mut self` 方法.
//!
//! # 视锥剔除
//!
//! [`Frustum`] 从视图投影矩阵中提取六个平面, 与 [`Aabb`] 组合即可完成
//! 剔除:
//!
//! ```ignore
//! let frustum = Frustum::from_view_projection(camera.view_projection());
//! render_list.retain_in_frustum(&frustum);
//! ```

pub mod aabb;
pub mod frustum;
pub mod orthographic;
pub mod perspective;

pub use aabb::Aabb;
pub use frustum::{Frustum, Plane};
pub use orthographic::OrthographicCamera;
pub use perspective::PerspectiveCamera;

use glam::{Mat4, Vec3};

use crate::system::Globals;

/// 相机查询接口.
///
/// 实现者只需实现 [`Self::view`] / [`Self::projection`] / [`Self::position`]
/// 与三个基向量, 其余方法 (组合矩阵, 提取视锥, 写入 [`Globals`]) 全部由
/// 默认实现从这些基本量导出.
///
/// # 只读接口
///
/// 本 trait 的方法都只接收 `&self` —— 相机是**只读查询对象**, 不含缓存
/// 或可变状态. 视口更新 (`set_viewport`) 由各实现单独提供, 不在本 trait
/// 的契约之内 —— 见模块文档的「视口变化」一节.
pub trait Camera {
	// ================ 必须实现的方法 ================

	/// 返回视图矩阵.
	///
	/// 把世界坐标变换到相机空间. 相机空间中相机位于原点, 朝向 `-Z`.
	fn view(&self) -> Mat4;

	/// 返回投影矩阵.
	///
	/// 把相机空间坐标变换到 NDC, 其中 NDC 的 `z` 落在 `[0, 1]` 内.
	fn projection(&self) -> Mat4;

	/// 返回相机在世界空间中的坐标.
	fn position(&self) -> Vec3;

	/// 返回相机朝向 (单位向量, 世界空间).
	///
	/// 相机看向 `-Z` —— 即世界空间中「屏幕朝向」所对应的方向.
	fn forward(&self) -> Vec3;

	/// 返回相机右方向 (单位向量, 世界空间).
	fn right(&self) -> Vec3;

	/// 返回相机上方向 (单位向量, 世界空间).
	fn up(&self) -> Vec3;

	// ================ 默认实现的方法 ================

	/// 返回视图投影矩阵.
	///
	/// 顶点着色器用 `view_projection * vec4(position, 1.0)` 即可算出
	/// 裁剪空间坐标.
	///
	/// # 顺序
	///
	/// 相乘顺序为 `projection * view` —— 先做视图变换, 再做投影, 与
	/// `glam` 列主序矩阵的乘法顺序一致.
	fn view_projection(&self) -> Mat4 {
		self.projection() * self.view()
	}

	/// 返回视图投影矩阵的逆矩阵.
	///
	/// 用于从 NDC 反投影回世界空间, 例如鼠标拾取与天空盒方向重建.
	fn inverse_view_projection(&self) -> Mat4 {
		self.view_projection().inverse()
	}

	/// 由视图投影矩阵提取视锥.
	///
	/// # 用途
	///
	/// 供剔除使用, 与 [`Aabb::intersects_frustum`] 组合:
	///
	/// ```ignore
	/// let frustum = camera.frustum();
	/// if aabb.intersects_frustum(&frustum) { /* 绘制 */ }
	/// ```
	///
	/// # 性能
	///
	/// 每次调用都会重新提取六个平面 (含 6 次 `sqrt` 归一化). 每帧调用
	/// 一次没有问题 —— 不要在逐对象循环里调用.
	fn frustum(&self) -> Frustum {
		Frustum::from_view_projection(self.view_projection())
	}

	/// 把相机状态写入全局 uniform.
	///
	/// # 行为
	///
	/// 会更新 [`Globals`] 的这些字段:
	///
	/// - `view_projection` —— 视图投影矩阵.
	/// - `view` —— 视图矩阵.
	/// - `inverse_view_projection` —— 视图投影矩阵的逆.
	/// - `camera_pos` —— `xyz` 分量为相机位置, `w` 分量置 1.0.
	///
	/// 其余字段 (`fog_color` / `fog_range` / `game_time` / `_pad`) 保持
	/// 不变 —— 交由调用方单独管理.
	fn apply(&self, globals: &mut Globals) {
		globals.view_projection = self.view_projection().to_cols_array_2d();
		globals.view = self.view().to_cols_array_2d();
		globals.inverse_view_projection =
			self.inverse_view_projection().to_cols_array_2d();

		let position = self.position();
		globals.camera_pos[0] = position.x;
		globals.camera_pos[1] = position.y;
		globals.camera_pos[2] = position.z;
		globals.camera_pos[3] = 1.0;
	}
}
