// SPDX-License-Identifier: MIT
// Copyright (c) 2026 CynFox Team

//! 顶点模块统一的错误枚举.
//!
//! 所有可能失败的公开 API 都以 [`Result<T, Error>`][] 返回, 调用方只需
//! 匹配这一个枚举. 变体按触发阶段分成两组: **格式构造**与**顶点写入**.
//!
//! [`Result<T, Error>`]: Result

use thiserror::Error;

use crate::vertex::element::{ElementType, ElementUsage, VertexFormatElement};

/// 顶点模块公开 API 可能返回的全部错误.
///
/// 枚举标注了 `#[non_exhaustive]`, 外部匹配必须包含 `_ => …`.
/// 字段只承载诊断信息, 不参与控制流 - 调用方应基于**变体**分支.
///
/// 需要 `wgpu::Device` 才能做的检查 (属性数量上限, `Features` 门控等)
/// **不在**本枚举范围内, 它们属于管线创建阶段, 由 `PipelineCache` 负责.
///
/// # 为什么只有一个 `Error`
///
/// 拆成多个子错误类型会迫使调用方:
///
/// - 分别 `use` 好几个类型;
/// - 在函数签名里出现多种 `Result<T, XxxError>`;
/// - 跨模块封装时额外写 `From` 转换.
///
/// 单枚举一次性消除了这些问题. 代价是变体较多, 但 `thiserror` 的
/// `#[error]` 让每个变体自带清晰的日志信息, 阅读成本主要落在定义处,
/// 而不是使用处.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum Error {
	// ================ 格式构造相关 ================
	//
	// 本组错误由 `VertexFormat::new` 及其依赖 (`VertexFormatElement::new`,
	// `VertexFormatElement::wgpu_format`) 返回. 它们说明「格式定义本身
	// 不合法」, 与顶点写入流程无关.

	/// 同一 usage 在格式里被声明了两次.
	///
	/// 非 UV 的 usage (`Position` / `Normal` / `Color` / `Padding` /
	/// `Generic`) 即使 `index` 不同也不允许重复; UV 则以 `uv_index`
	/// 为键判重. 由 [`VertexFormat::new`][] 返回.
	///
	/// # 与 `element.rs` 的关系
	///
	/// `VertexFormatElement::new` 已经拒绝了非 UV 用途的非零 `index`,
	/// 所以这里遇到的重复只有两种:
	///
	/// - 两个 usage 相同的非 UV 元素 (index 都是 0);
	/// - 两个 `uv_index` 相同的 UV 元素.
	///
	/// **不要**放宽这里的判重逻辑, 否则着色器端的 `@location(n)` 会与
	/// 元素列表对不上.
	///
	/// [`VertexFormat::new`]: crate::vertex::format::VertexFormat::new
	#[error("duplicate element usage {usage:?}")]
	DuplicateUsage {
		/// 引发冲突的用途. UV 冲突时恒为 [`ElementUsage::Uv`][].
		usage: ElementUsage,
	},

	/// 格式里出现了两个同名元素.
	///
	/// 名字供调试与 [`index_of_name`][] 使用, 重名会让按名查下标产生
	/// 歧义. 由 [`VertexFormat::new`][] 返回.
	///
	/// # 注意
	///
	/// 名字**不参与** `VertexFormat` 的 `PartialEq` / `Hash` 比较, 因此
	/// 重名不会影响管线缓存. 这里报错只是为了让 `index_of_name` 的行为
	/// 确定.
	///
	/// [`index_of_name`]: crate::vertex::format::VertexFormat::index_of_name
	/// [`VertexFormat::new`]: crate::vertex::format::VertexFormat::new
	#[error("Duplicate element name {name:?}")]
	DuplicateName {
		/// 出现了两次的元素名.
		name: String,
	},

	/// 格式里连一个元素都没有.
	///
	/// 空格式既算不出 stride, 也无法声明任何属性. 由
	/// [`VertexFormat::new`][] 返回.
	///
	/// # 为什么必须拒绝
	///
	/// wgpu 会拒绝 `array_stride == 0` 的 `VertexBufferLayout`. 与其等到
	/// 管线创建时触发 wgpu 的 panic, 不如在这里就明确报错.
	///
	/// [`VertexFormat::new`]: crate::vertex::format::VertexFormat::new
	#[error("vertex format has no elements")]
	Empty,

	/// 格式的第一个元素是 `Padding`.
	///
	/// `Building` 的写入流程假定第一个元素是可写入的属性 —— 前导的
	/// `Padding` 会让 `current_element` 停在 Padding 上, 第一次调用
	/// `vertex()` 就以 [`Error::WrongElement`][] 失败.
	///
	/// 中间与末尾的 `Padding` 是合法的 (用于对齐), 只有开头不允许.
	/// 由 [`VertexFormat::new`][] 返回.
	///
	/// # 为什么不在 `Building` 里跳过
	///
	/// 跳过前导 `Padding` 需要在 `advance` 里区分两种情形: 「首次从 0
	/// 出发」与「写完一个顶点后从末尾回到 0」—— 后者不应再次跳过前导
	/// 的 `Padding`. 与其在热路径上增加分支, 不如在构造时直接拒绝.
	///
	/// [`VertexFormat::new`]: crate::vertex::format::VertexFormat::new
	#[error("vertex format must not start with Padding")]
	LeadingPadding,

	/// 元素的 `count` 为 0, 占用的字节数也是 0.
	///
	/// 这种元素不贡献任何字节却仍占一个槽位, 会让 `Building` 的
	/// `next_element_byte` 停在原地, 破坏顺序语义. 由
	/// [`VertexFormat::new`][] 返回.
	///
	/// # 为什么是致命错误
	///
	/// `Building` 依靠 `byte_size > 0` 来保证每写一个元素游标都向前
	/// 推进. 一旦出现零大小元素, 顶点内的位置原地不动, 后续写入就会
	/// 覆盖同一段内存, 造成静默的数据损坏.
	///
	/// [`VertexFormat::new`]: crate::vertex::format::VertexFormat::new
	#[error("element {element:?} has zero size")]
	ZeroSizedElement {
		/// 字节数为 0 的那个元素.
		element: VertexFormatElement,
	},

	/// 元素找不到对应的 [`wgpu::VertexFormat`][].
	///
	/// 例如 `(Uv, Float, 3)` 或 `(Normal, UByte, 2)`. `Padding` 不会
	/// 触发该错误 - 它在 `wgpu_format()` 里返回 `Ok(None)`.
	///
	/// # 与 `element.rs` 的关系
	///
	/// 该错误由 [`VertexFormatElement::wgpu_format`][] 返回, 并在
	/// `VertexFormat::new` 中通过 `?` 提前传播. 在构造阶段就拦下它,
	/// 可以避免等到创建管线时才发现格式不可用.
	///
	/// [`wgpu::VertexFormat`]: wgpu::VertexFormat
	/// [`VertexFormatElement::wgpu_format`]: VertexFormatElement::wgpu_format
	#[error("element {element:?} has no wgpu equivalent")]
	UnsupportedElement {
		/// 无法映射到 wgpu 格式的元素.
		element: VertexFormatElement,
	},

	/// 格式的总字节数 (stride) 越过了 `u32::MAX`.
	///
	/// 上游把 `offset` 与 `stride` 统一表达为 `u32`, 避免和 `usize` 互转
	/// 时发生静默截断. 正常格式的 stride 在 4 到 64 字节之间, 能触发此
	/// 错误, 几乎总是因为格式定义本身有误.
	///
	/// # 为什么上限是 `u32` 而不是 `u64`
	///
	/// wgpu 的 `VertexAttribute::offset` 与 `VertexBufferLayout::array_stride`
	/// 在 API 层面都是 `BufferAddress` (u64), 不过:
	///
	/// - 有些后端 (D3D, 移动 GPU) 对顶点缓冲的寻址限于 32 位.
	/// - 上游代码 (`BufferBuilder`, `DrawState`) 统一用 `u32`, 保持类型一致.
	///
	/// 在 `new` 阶段就拦住溢出, 总比等到渲染时崩溃更容易定位.
	#[error("vertex stride would overflow u32")]
	StrideOverflow,

	/// UV 以外的用途带了非零 `index`.
	///
	/// 由 [`VertexFormatElement::new`][] 返回. UV 用途的 `index` 表示 UV
	/// 层号, 允许非零; 其他用途在格式中只应出现一份, `index` 恒为 0.
	///
	/// # 与 `format.rs` 的关系
	///
	/// 这项校验必须与 `VertexFormat::new` 中「非 UV 用途按 usage 判重」
	/// 的逻辑**保持一致**. 一旦这里放宽, 准许非 UV 用途携带非零
	/// `index`, `format.rs` 仍会把它们当成重复, 于是构造单个元素时
	/// 合法, 组合成格式后却报错, 诊断信息自相矛盾.
	///
	/// [`VertexFormatElement::new`]: VertexFormatElement::new
	#[error("usage {usage:?} must have index 0, got {index}")]
	InvalidIndex {
		/// 触发错误的用途.
		usage: ElementUsage,
		/// 调用方传入的非零索引.
		index: u32,
	},

	// ================ 顶点写入相关 ================
	//
	// 本组错误由 `VertexConsumer` 的实现 (主要是 `Building`) 在写入
	// 过程中返回. 它们说明「格式本身没问题, 出问题的是写入流程」.

	/// 当前元素与调用方法期望的元素不一致.
	///
	/// [`VertexConsumer`][] 按格式顺序逐个元素推进, 每个写入方法都要求
	/// 当前元素与自身语义相符, 不符就报错.
	///
	/// **常见成因**: 调用顺序与格式声明顺序不一致; 格式与调用约定
	/// 不匹配 (例如向 `position_tex` 写入 color); 写入流程漏掉了某个
	/// 元素.
	///
	/// 对照 `actual` 与 `expected_*` 可以直接定位错位的位置.
	///
	/// # 与 `format.rs` 的关系
	///
	/// `actual` 取自 `Building` 内部的 `current_element`, 而它由
	/// `VertexFormat::elements` 的顺序决定. 调试时把两者与格式定义
	/// 对照, 就能看出是哪一环写错了.
	///
	/// [`VertexConsumer`]: crate::vertex::consumer::VertexConsumer
	#[error("expected {expected_usage:?} ({expected_type:?} x{expected_count}), got {actual:?}")]
	WrongElement {
		/// 写入流程当前停留的元素.
		actual: VertexFormatElement,
		/// 调用方法所要求的用途.
		expected_usage: ElementUsage,
		/// 调用方法所要求的标量类型.
		expected_type: ElementType,
		/// 调用方法所要求的分量数.
		expected_count: u32,
	},

	/// 顶点还没写满就调用了 `end_vertex`.
	///
	/// `Building` 依据「当前顶点已写入的字节数等于 stride」来判断一个
	/// 顶点是否写满. 状态对不上, 说明本轮写入流程提前结束 - **漏调了
	/// 某个写入方法**, 或者中途用 `?` / `continue` 提前退出.
	///
	/// `remaining == total` 表示一个元素都还没写.
	///
	/// # 与 `format.rs` 的关系
	///
	/// `total` 等于 `VertexFormat::vertex_size()`, `remaining` 则由
	/// `Building` 的 `next_element_byte` 与 `vertices` 算出. 两者单位
	/// 都是字节, 可以直接和格式定义对照.
	#[error("vertex incomplete: {remaining} of {total} bytes written")]
	IncompleteVertex {
		/// 一个顶点占用的总字节数.
		total: u32,
		/// 距离写满还缺少的字节数.
		remaining: u32,
	},

	/// 默认颜色生效期间, 不允许再手动调用 `color`.
	///
	/// 调用 [`default_color`][] 之后, `Color` 元素由 `Building` 自动
	/// 填充. 此时若再显式调用 `color`, 就会出现「默认值与显式值谁
	/// 生效」的歧义. 与其在运行时替调用方二选一, 不如直接报错, 要求
	/// 意图明确:
	///
	/// - 想用默认值: 不要调用 `color`.
	/// - 想用显式值: 先调用 [`unset_default_color`][].
	///
	/// # 与 `element.rs` 的关系
	///
	/// 只有格式里存在 `Color` 元素时, 该错误才可能发生. 若格式不含
	/// `Color`, `default_color` 依然可以设置, 但 `Building::advance`
	/// 在跳转时不会填充它 —— 相当于无声失效. 这一点在 `Building` 的
	/// 实现处有注释说明.
	///
	/// [`default_color`]: crate::vertex::consumer::VertexConsumer::default_color
	/// [`unset_default_color`]: crate::vertex::consumer::VertexConsumer::unset_default_color
	#[error("cannot manually set color while a default color is active")]
	DefaultColorActive,

	/// 顶点数量超过 `u32::MAX`.
	///
	/// 这是一道防御性检查: 避免计数器回绕后引发难以追踪的下游故障.
	/// 触发它几乎总意味着上层逻辑有误 (循环没能正确终止, 顶点数据被
	/// 重复处理), 应当在调用方修正, 而不是放宽上限.
	///
	/// # 与 `format.rs` 的关系
	///
	/// `Building::vertices` 的类型是 `u32`, 与 `DrawState::vertex_count`
	/// 保持一致. 若把 `u32` 改成 `u64`, 整条链路都要跟着改, 并不
	/// 划算 —— 单个批次达到 40 亿顶点本身就不现实.
	#[error("vertex count overflow")]
	VertexOverflow,
}

/// 顶点模块使用的 `Result` 别名.
///
/// 便于把函数签名写成 `Result<T>`; 需要时也可以用
/// `Result<T, OtherError>` 覆盖默认的错误类型.
///
/// # 与 crate 顶层 `Result` 的关系
///
/// 即便 crate 顶层将来引入自己的 `Result`, 本别名也不受影响 ——
/// 调用方通过 `use crate::vertex::error::Result` 明确取用顶点模块的
/// 版本.
pub type Result<T, E = Error> = std::result::Result<T, E>;
