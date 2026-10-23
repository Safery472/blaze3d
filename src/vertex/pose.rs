//! 按层级组织的变换栈.
//!
//! [`PoseStack`][] 内部维护一组 `(pose, normal)` 矩阵对: `pose` 负责变换
//! 顶点位置, `normal` 负责变换法线, 两者必须同步更新.
//!
//! # 为什么需要两个矩阵
//!
//! 一旦出现非均匀缩放, 顶点位置与法线就要用不同的变换矩阵: 顶点用
//! 正向变换矩阵, 法线用「逆矩阵的转置」. 若对法线直接套用正向矩阵,
//! 法线会偏离表面.
//!
//! [`PoseStack`][] 用近似公式代替精确求逆:
//!
//! - 均匀缩放: 只调整符号, 不改变长度.
//! - 非均匀缩放: 缩放到归一化行列式, 以保持光照强度.
//!
//! 该近似的精度对光照而言已经足够, 同时省去了每步求逆的开销.
//!
//! # 用法
//!
//! 显式调用 `push` / `pop`, 由调用方手动配对:
//!
//! ```rust
//! use blaze3d::vertex::PoseStack;
//!
//! let mut stack = PoseStack::new();
//! stack.push();
//! stack.translate(1.0, 0.0, 0.0);
//! stack.pop();
//!
//! assert!(stack.is_clear());
//! ```
//!
//! # RAII guard
//!
//! [`PoseStack::scoped`][] 返回 [`PoseGuard`][], 后者在 `drop` 时自动执行
//! `pop`. 与手动 `push` / `pop` 相比, 它在 panic 与提前返回的情形下
//! 同样不会破坏栈平衡.
//!
//! **注意**: `PoseGuard` 借用了整个 `PoseStack` —— guard 存活期间, 栈的
//! 任何方法 (包括 `translate` / `rotate` 等变换) 都无法调用. 这与
//! 「进入一段作用域, 在内部修改变换」的直觉并不一致.
//!
//! 需要在作用域内修改变换时, 应改用显式 `push` / `pop`:
//!
//! ```rust
//! use blaze3d::vertex::PoseStack;
//!
//! let mut stack = PoseStack::new();
//! stack.push();
//! stack.translate(1.0, 2.0, 3.0);
//! stack.pop();
//!
//! assert!(stack.is_clear());
//! ```
//!
//! [`PoseGuard`][] 适用于「只需保证 `pop` 时机, 内部不改动栈」的场合 ——
//! 例如把栈的访问权限临时转交给另一个函数.

use glam::{Mat3, Mat4, Quat, Vec3};

/// 一对相互关联的变换矩阵.
///
/// 其中 `pose` 负责顶点位置, `normal` 负责法线, 两者始终一起更新.
#[derive(Debug, Clone, Copy)]
pub struct Pose {
	/// 作用于顶点位置的变换矩阵.
	pub pose: Mat4,
	/// 作用于法线的变换矩阵.
	pub normal: Mat3,
}

impl Pose {
	/// 返回单位变换, 即 `pose` 与 `normal` 都取单位矩阵.
	pub const fn identity() -> Self {
		Self {
			pose: Mat4::IDENTITY,
			normal: Mat3::IDENTITY,
		}
	}
}

/// 按层级组织的变换栈.
///
/// 栈在初始状态就含一个单位 [`Pose`][], 所有变换都作用在栈顶元素上.
///
/// # 设计要点
///
/// - **栈永不为空**: 初始就有一个元素, 当栈中只剩它时 `pop` 会 panic.
///   这与「总存在一个变换基底」的语义一致.
/// - **`normal` 与 `pose` 同步**: 每个变换方法都会同时更新两者, 不应
///   只改其中之一.
#[derive(Debug, Clone)]
pub struct PoseStack {
	// 栈底存放单位 `Pose`, 它永远不会被弹出.
	// 选用 `Vec` 而不是 `VecDeque`: 操作只发生在两端, `Vec` 的缓存局部性
	// 更佳.
	stack: Vec<Pose>,
}

impl Default for PoseStack {
	fn default() -> Self {
		Self::new()
	}
}

impl PoseStack {
	/// 新建一个只含单位变换的栈.
	pub fn new() -> Self {
		Self {
			stack: vec![Pose::identity()],
		}
	}

	// ================ 入栈 / 出栈与查询 ================

	/// 复制栈顶元素并压入栈.
	///
	/// 新元素持有 `pose` 与 `normal` 的独立副本, 改动它不会波及原来的
	/// 栈顶.
	pub fn push(&mut self) {
		// 不变量: 栈非空. 该不变量由 `new` 建立, `pop` 也不会把栈删到空.
		let top = *self.stack.last().expect("stack is never empty");
		self.stack.push(top);
	}

	/// 弹出栈顶元素.
	///
	/// # Panics
	///
	/// 当栈深度为 1 时 panic. 需要避免 panic 时改用 [`Self::scoped`][].
	pub fn pop(&mut self) {
		assert!(self.stack.len() > 1, "PoseStack underflow");
		self.stack.pop();
	}

	/// 返回一个 RAII guard, 它在 `drop` 时自动调用 [`Self::pop`][].
	///
	/// # 借用语义
	///
	/// guard 借走了整个 `PoseStack`, 因此在它存活期间无法调用栈的任何
	/// 方法. 这是由 `&mut self` 方法的性质带来的固有冲突: guard 一直
	/// 持有独占借用, 直到自身被 `drop`.
	///
	/// # 用途
	///
	/// 适合「只需保证 `pop` 时机」的场合 —— 例如把栈临时借给另一段代码
	/// 使用:
	///
	/// ```rust
	/// use blaze3d::vertex::PoseStack;
	///
	/// fn with_stack(stack: &mut PoseStack) {
	///     let _guard = stack.scoped();
	///     // guard 存活期间栈被独占借用 —— 这里不能访问 stack.
	///     // 但 guard drop 时自动 pop, 无需手动配对.
	/// }
	///
	/// let mut stack = PoseStack::new();
	/// with_stack(&mut stack);
	/// assert!(stack.is_clear());
	/// ```
	///
	/// 需要在作用域内执行变换时, 请改用显式 `push` / `pop`.
	pub fn scoped(&mut self) -> PoseGuard<'_> {
		self.push();
		PoseGuard { stack: self }
	}

	/// 返回栈顶的矩阵对.
	pub fn last(&self) -> Pose {
		*self.stack.last().expect("stack is never empty")
	}

	/// 返回栈顶的 `pose` 矩阵.
	pub fn top(&self) -> Mat4 {
		self.last().pose
	}

	/// 返回栈顶的 `normal` 矩阵.
	pub fn top_normal(&self) -> Mat3 {
		self.last().normal
	}

	/// 返回当前栈深度 (元素个数), 初始值为 1.
	pub fn depth(&self) -> usize {
		self.stack.len()
	}

	/// 判断栈中是否只剩最初压入的单位元素.
	pub fn is_clear(&self) -> bool {
		self.stack.len() == 1
	}

	/// 清空栈, 重置为只含单位矩阵的状态.
	pub fn reset(&mut self) {
		self.stack.clear();
		self.stack.push(Pose::identity());
	}

	// ================ 变换与状态改写 ================
	//
	// 以下变换都作用于栈顶, 并同步更新 `pose` 与 `normal`.

	/// 按给定偏移平移.
	///
	/// 该操作只改写 `pose`. 平移不改变朝向, 因此 `normal` 保持原样.
	pub fn translate(&mut self, x: f32, y: f32, z: f32) -> &mut Self {
		let top = self.stack.last_mut().expect("stack is never empty");
		top.pose *= Mat4::from_translation(Vec3::new(x, y, z));
		// `normal` 无需改动.
		self
	}

	/// 按三个轴分别缩放.
	///
	/// # 法线处理
	///
	/// - **均匀且正缩放** (`x == y == z && x > 0`): `normal` 不变.
	///   均匀正缩放既不改变法线方向, 也不改变其长度.
	/// - **均匀且非正缩放** (`x == y == z && x <= 0`): 先把 `normal` 的
	///   朝向翻转, 然后**继续**进入非均匀分支 —— 这两步的效果会叠加.
	///   翻转用于反映镜像, 归一化用于保持光照强度.
	/// - **非均匀缩放**: 采用近似公式 `k = (1/(x*y*z))^(1/3)`, 让
	///   `normal` 乘以 `(k/x, k/y, k/z)`. 这样 `normal` 的行列式保持为
	///   1, 光照强度不随缩放变化.
	///
	/// 上述做法是近似, 并非精确的「逆矩阵的转置」. 精确做法每一步都要
	/// 求矩阵的逆, 而视觉上的收益不足以抵消这份开销.
	///
	/// # 注意
	///
	/// **均匀非正缩放不是提前返回**: 翻转之后仍会走非均匀分支. 若把它
	/// 改成 `return self`, 负缩放的 `normal` 就会缺少归一化步骤.
	///
	/// # 为什么用 `Mat3::from_diagonal` 而不是 `from_scale`
	///
	/// `glam` 的 `Mat3::from_scale` 接收 `Vec2`, 语义是 2D 齐次缩放.
	/// 此处需要的是 3D 对角缩放, 必须用 `Mat3::from_diagonal(Vec3)`.
	/// 两者视觉上相似, 但前者只作用于 xy, z 分量固定为 1.
	pub fn scale(&mut self, x: f32, y: f32, z: f32) -> &mut Self {
		let top = self.stack.last_mut().expect("stack is never empty");
		top.pose *= Mat4::from_scale(Vec3::new(x, y, z));

		// 先处理均匀缩放这一特例.
		if x == y && y == z {
			if x > 0.0 {
				// 均匀正缩放: `normal` 无需改动, 可直接返回.
				return self;
			}
			// 均匀非正缩放: 翻转 `normal` 的朝向.
			// **此处不可** return —— 后面还有归一化步骤要执行.
			top.normal *= Mat3::from_diagonal(Vec3::splat(-1.0));
		}

		// 非均匀缩放 (也包括均匀非正缩放的归一化部分).
		//
		// 归一化公式为 k = (1/(x*y*z))^(1/3), 即让 `normal` 乘以
		// (k/x, k/y, k/z). 所得矩阵的行列式为 1, 只改变方向, 不改变
		// 「体积」.
		//
		// x, y, z 中任一为 0 时, 结果会是 NaN / Inf —— 这与上游行为
		// 一致. 缩放分量非零由调用方自行保证.
		let inv = 1.0 / (x * y * z);
		let k = inv.cbrt();

		top.normal *= Mat3::from_diagonal(Vec3::new(k / x, k / y, k / z));
		self
	}

	/// 用四元数对栈顶做旋转.
	///
	/// 旋转对 `pose` 与 `normal` 的作用相同 —— 旋转属于正交变换, 不会
	/// 改变长度.
	pub fn rotate(&mut self, q: Quat) -> &mut Self {
		let top = self.stack.last_mut().expect("stack is never empty");
		top.pose *= Mat4::from_quat(q);
		top.normal *= Mat3::from_quat(q);
		self
	}

	/// 绕过给定点的轴旋转.
	///
	/// 效果等价于 `translate(p) * rotate(q) * translate(-p)`: 先把旋转
	/// 中心平移到原点, 旋转完成后再平移回去.
	///
	/// `normal` 只需施加旋转 —— 平移不会改变法线.
	pub fn rotate_around(&mut self, q: Quat, x: f32, y: f32, z: f32) -> &mut Self {
		let top = self.stack.last_mut().expect("stack is never empty");
		let pivot = Vec3::new(x, y, z);
		top.pose *= Mat4::from_translation(pivot)
			* Mat4::from_quat(q)
			* Mat4::from_translation(-pivot);
		top.normal *= Mat3::from_quat(q);
		self
	}

	/// 在栈顶 `pose` 的右侧乘上任意矩阵.
	///
	/// # 注意
	///
	/// **只改写 `pose`**, 不会更新 `normal`. `normal` 的一致性由调用方
	/// 自行负责 —— 传入旋转矩阵时, 通常还需要同步旋转 `normal`. 正因
	/// 如此, 该方法是低层接口, 而不是推荐用法.
	pub fn mul_matrix(&mut self, m: Mat4) -> &mut Self {
		let top = self.stack.last_mut().expect("stack is never empty");
		top.pose *= m;
		self
	}

	/// 用给定的矩阵对直接覆盖栈顶.
	///
	/// 该操作跳过所有计算, 适合从缓存中恢复状态.
	pub fn set_pose(&mut self, pose: Mat4, normal: Mat3) -> &mut Self {
		let top = self.stack.last_mut().expect("stack is never empty");
		top.pose = pose;
		top.normal = normal;
		self
	}

	/// 把栈顶的矩阵对重置为单位矩阵.
	pub fn set_identity(&mut self) -> &mut Self {
		let top = self.stack.last_mut().expect("stack is never empty");
		top.pose = Mat4::IDENTITY;
		top.normal = Mat3::IDENTITY;
		self
	}

	// ================ 轴向旋转快捷方法 ================

	/// 绕 X 轴旋转给定弧度.
	pub fn rotate_x(&mut self, radians: f32) -> &mut Self {
		self.rotate(Quat::from_rotation_x(radians))
	}

	/// 绕 Y 轴旋转给定弧度.
	pub fn rotate_y(&mut self, radians: f32) -> &mut Self {
		self.rotate(Quat::from_rotation_y(radians))
	}

	/// 绕 Z 轴旋转给定弧度.
	pub fn rotate_z(&mut self, radians: f32) -> &mut Self {
		self.rotate(Quat::from_rotation_z(radians))
	}

	/// 绕 X 轴旋转给定角度 (单位为度).
	pub fn rotate_x_deg(&mut self, degrees: f32) -> &mut Self {
		self.rotate_x(degrees.to_radians())
	}

	/// 绕 Y 轴旋转给定角度 (单位为度).
	pub fn rotate_y_deg(&mut self, degrees: f32) -> &mut Self {
		self.rotate_y(degrees.to_radians())
	}

	/// 绕 Z 轴旋转给定角度 (单位为度).
	pub fn rotate_z_deg(&mut self, degrees: f32) -> &mut Self {
		self.rotate_z(degrees.to_radians())
	}

	/// 绕任意轴旋转给定角度 (单位为度).
	///
	/// 传入的 `axis` 会先被归一化. 传零向量时的行为取决于 `glam` 的
	/// 实现.
	pub fn rotate_axis_deg(&mut self, axis: Vec3, degrees: f32) -> &mut Self {
		self.rotate(Quat::from_axis_angle(
			axis.normalize(),
			degrees.to_radians(),
		))
	}
}

/// 由 [`PoseStack::scoped`][] 返回的 RAII guard.
///
/// 它会在 `drop` 时自动调用 [`PoseStack::pop`][], 从而保证栈平衡.
///
/// # 借用语义
///
/// guard 内部持有 `&mut PoseStack`, 因此它存活期间无法再访问栈. 这是
/// 有意的设计: 若让 guard 提供变换方法, 就必须把 `PoseStack` 的全部
/// API 都转发一遍, 代价大于收益.
pub struct PoseGuard<'a> {
	stack: &'a mut PoseStack,
}

impl Drop for PoseGuard<'_> {
	fn drop(&mut self) {
		self.stack.pop();
	}
}
