# Changelog

本文件记录 blaze3d 所有重要变更, 按版本倒序排列.

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/),
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/).

## [Unreleased]

## [0.2.0] - 2026-11-01

一次以**正确性**为主的版本: 修掉 15 项审计发现, 补齐阴影管线缺失的两块
(`CubeShadowMap` 接入 `Target`, 蒙皮/网格阴影着色器), 并把 `.min.wgsl`
换成真正的压缩产物.

### 破坏性变更

升级前请对照本小节; 其余条目均不影响现有调用方.

- `Target::color_attachment` 返回 `Option<wgpu::RenderPassColorAttachment<'_>>`
  —— 纯深度 pass 不再需要挂一个随即丢弃的 dummy 颜色附件. 自行实现
  `Target` 的代码需把返回值包成 `Some(...)`
- `RenderType::color_format` 与 `RenderTypeBuilder` 的同名字段改为
  `Option<ColorFormat>`. 阴影等纯深度管线应改用
  `RenderTypeBuilder::depth_only()`; 配 `RenderTarget` / `SurfaceFrame`
  的管线照常调用 `.color_format(...)` 即可
- `MipmapGenerator::generate` 在 `mip_count` 之后新增
  `base_size: (u32, u32)`. 原实现调用了 `wgpu::Texture` 上不存在的
  `width()` / `height()`, 本身无法编译
- `formats::block` 的末尾 `Padding` 由 `UByte x1` 改为 `UByte x4`, 两个
  预设的 stride 因此由 33 / 37 变为 36 / 40. 手写顶点字节的代码需按新
  stride 调整
- `ObjectUniforms` 由 64 字节增至 80 字节 (新增 `bone_start` /
  `bone_count`). 自定义着色器若复用了该结构需同步补字段

### 修复

- `BoneTrack::new` 拒绝 NaN, 无穷与负的关键帧时间 (新增
  `Error::TrackInvalidTime`); `BoneTrack::sample` 改用 `checked_sub`,
  绕过构造函数构造出的畸形轨道不再 panic
- Cube 维度的 mipmap 之前只复制源级左上角纹素, 现在按 2x2 抽头取平均:
  `shaders/mipmap_cube.*.wgsl` 更名为 `mipmap_array.*.wgsl` (可读版与
  压缩版同步)
- 整数 storage 格式 (`Rgba8Uint` 等) 的 mipmap 之前生成必然失败的浮点
  着色器, 现在走整数的专用着色器: 新增 `shaders/mipmap_integer*.wgsl`
  (2D / 3D / 2D 数组), layout 采样类型与 bind group 条目一并切换
- `Texture::from_bytes` 拒绝压缩, 深度与模板格式 (`UnsupportedFormat`),
  缺 `COPY_DST` 时返回 `MissingCopyDst`; 之前会在 `write_texture` 处
  产生 wgpu 验证错误
- `Texture::new` / `from_bytes` 校验 Mip 层数上限
  (`Error::InvalidMipCount`), `MipmapGenerator::generate` 同样校验并
  拒绝 MSAA 纹理
- `VertexFormatElement::byte_size` 不再溢出回绕: 新增
  `checked_byte_size`, `VertexFormat::new` 用它返回 `StrideOverflow`
- `(Normal, Byte, 3)` 按 4 字节槽位计入 stride (wgpu 侧为 `Snorm8x4`),
  写入时第 4 个分量补 0; `VertexFormat::new` 新增 stride 4 字节对齐校验
  (`Error::StrideMisaligned`)
- `ShadowMap::new` 的正交投影改为 `near = radius * 0.5`,
  `far = radius * 3.5`; 之前 `near` 固定 0.1, 半径小于 0.025 的场景
  深度会被映射到 `[0, 1]` 之外
- `read_texture` 校验单采样与 `COPY_SRC` (`ReadbackUnsupportedSampleCount` /
  `ReadbackMissingSource`), 行宽计算改用 `u64`; `read_buffer` 校验
  `size` 的对齐与容量 (`ReadbackInvalidSize`) 以及 `COPY_SRC`
- `configure_surface` 拒绝零尺寸 (`Error::InvalidSurfaceSize`)
- `Direction::nearest` 的零向量返回 `Up`, 与文档一致
- `shaders/mipmap_array.wgsl` 的累加结果改用 `sum / {VTYPE}(4)`, 与
  `mipmap_array_integer.wgsl` 一致: 原来的 `sum * 0.25` 是抽象浮点
  字面量, 在整数格式下要求隐式转换

### 新增

- `CubeShadowMap::render_face` 与 `CubeFaceTarget`: 立方体阴影的单个面
  直接实现 `Target`, 可以交给 `RenderSystem::render_frame`
- `shaders/shadow_depth_mesh.wgsl` / `shadow_depth_skinned.wgsl` 与对应的
  `BuiltinShader::ShadowDepthMesh` / `ShadowDepthSkinned` /
  `ShadowDepthInstanced`; `BuiltinShader::ALL` 由 5 项增至 8 项
- `RenderTypeBuilder::depth_only`: 构造纯深度管线
- `ObjectBuffer::push_skinned` 与 `RenderSystem::push_object_skinned`:
  随对象写入骨骼范围
- `PostProcessChain` / `PostProcessGraph` 的 `with_input_bindings` 与
  `InputBindings`, 支持 texture / sampler 不在 binding 0 / 1 的 layout
- `ShadowMap::light_view_projection_for`: 按给定场景包围盒与光照方向
  计算光空间视图投影
- `set_present_mode`: 选择呈现模式并返回实际生效的值
- `VertexFormatElement::checked_byte_size` 与 `padding_u8x4`
- 测试 (24 项): `tests/audit_regressions.rs` 覆盖骨骼轨道, 顶点格式,
  法线写入偏移, 朝向, 阴影投影与 Mip 层数上限; `tests/shader_compile.rs`
  在真实设备上编译全部内嵌着色器, 并校验可读版与压缩版成对存在;
  `tests/shadow_pass.rs` 覆盖立方体面渲染与三种阴影着色器;
  `tests/vertex_writes.rs` 覆盖法线写入的 4 字节槽位
- `tools/wgsl-min`: 调用 [`nagami`](https://github.com/ekarad1um/Nagami)
  生成 `.min.wgsl` 的薄封装, 负责遍历目录, 替换模板占位符, 校验两条
  探针产物一致, 并提供 `--check` 模式

### 删除

- `examples/audit_cpu` 与 `examples/audit_gpu`: 审计期的临时探针, 结论已
  固化为上述测试, 探针本身不再保留

### 变更

- 全部 17 个 `shaders/*.min.wgsl` 改由 `nagami` 生成: 之前只是"剥注释 +
  折叠空白", 仍是可读源码; 现在会降为 naga IR 做死代码消除, 常量折叠与
  内联, 标识符缩写, 输出为单行. 合计 14071 -> 9679 字节

### 重构

- `Frame` 的时间线写入不再重复 `unwrap`, 收敛为 `write_pass_begin_timestamp` /
  `write_pass_end_timestamp`
- `PoseStack` 的 9 处 `expect("stack is never empty")` 收敛到
  `top_or_identity` / `top_mut`
- `MipmapGenerator` 不再自建全屏四边形, 复用 `post::FullscreenQuad`
- `shadow` / `cube` 各一份的 `look_at_rh` 合并到 `pipeline::transform`
- 修正过时文档:「Mipmap 只支持 D2」, `Target` 里指向错误模块的
  `Error::ZeroSize`, 以及失效的 `StrideMisaligned` 文档链接

### 已知问题

- 蒙皮阴影的骨骼范围随 `ObjectUniforms` 走 group 1; 内置的
  `position_color_skinned.wgsl` 仍从 `Instance` 结构体的
  `bone_offset` / `bone_count` 取这两项, 两条路径的骨骼来源不同

### 其他

- 新增 CHANGELOG.md, 按 Keep a Changelog 格式记录本库变更
- 纪念某只意外上吊的鼠鼠. 愿它在没有渲染管线的地方安好.

## [0.1.2] - 2026-10-28

### 新增

- `SurfaceFrame::set_clear_color` / `clear_color`, 以及 `DEFAULT_SURFACE_CLEAR_COLOR`
  (不透明黑), 与 `RenderTarget::clear_color` 语义一致
- `examples/common`: `Gfx` / `Scene` / `Runner` 共用骨架, 统一 wgpu 初始化,
  surface 配置, resize 处理, MSAA 与深度附件重建, 时间管理
- 示例 `cube3d`: 单 pass 彩色立方体, 4x MSAA 与垂直同步
- 示例 `rainbow_cube`: 12 条渐变圆柱边 + 8 个角球, 边内线性渐变, 透明度闪烁
- crate 级 `#![recursion_limit = "256"]`, lib 与每个 example 各自声明

### 变更

- `Target::color_attachment` 改为读取 `SurfaceFrame::clear_color`, 默认值仍为
  不透明黑, 现有行为不变
- 示例的旋转角速度, 边宽, 闪烁深度与速度, 背景色均支持编译时环境变量覆盖,
  每个示例使用独立前缀 (`CUBE3D_*` / `RAINBOW_*`)

## [0.1.1] - 2026-10-24

### 变更

- Rust 代码添加 SPDX 许可证标识 (MIT)
- WGSL 添加 min 版本, 减小 `include_str!` 嵌入二进制的源码体积

## [0.1.0] - 2026-10-23

首个公开版本. 基于 wgpu 的 Rust 3D 渲染库, 面向方块/体素场景.

### 新增

- 相机与视锥剔除: `PerspectiveCamera` / `OrthographicCamera`, `Aabb`
- 顶点构建: `VertexFormat`, `BufferBuilder` / `Building` 状态机, `GpuMesh`
- 纹理与 mipmap: 2D / 3D / Cube 纹理, 采样器, compute shader 生成 mipmap
- 管线状态与缓存: `RenderTypeBuilder`, `PipelineCache`
- 渲染目标: `RenderTarget`, `SurfaceFrame`, `DepthBuffer` / `MsaaBuffer`
- 阴影: `ShadowMap`, `CubeShadowMap`
- 后处理: `PostProcessChain`, `PostProcessGraph`
- 渲染系统: `RenderSystem`, `Frame` / `PassHelper`, 实例化, 蒙皮, GPU 时间线
- GPU 读回: `read_texture` / `read_target` / `read_buffer`
- 骨骼动画: `Skeleton`, `SkeletonPose`, `AnimationClip`, `BoneTrack`

[Unreleased]: https://github.com/Safery472/blaze3d/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Safery472/blaze3d/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/Safery472/blaze3d/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Safery472/blaze3d/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Safery472/blaze3d/releases/tag/v0.1.0
