# blaze3d

基于 [wgpu](https://wgpu.rs) 的 Rust 3D 渲染库, 面向方块 / 体素场景. 从相机, 顶点构建, 纹理与 mipmap, 管线状态缓存, 阴影贴图, 后处理一直到 GPU 读回, 都提供开箱可用的实现.

> **商标声明**：“Minecraft”是 Mojang Synergies AB 和 Microsoft Corporation 的商标。`blaze3d` 是独立开源项目，与 Mojang 或 Microsoft 无隶属关系，也未获其背书。文中提及“Minecraft 风格”仅用于描述本库的适用场景，不表示任何官方关联或兼容性认证。

crate 级别禁止 `unsafe` (`#![deny(unsafe_code)]`), 所有着色器以 WGSL 源码形式内嵌, 不依赖运行时文件.

## 功能

- **相机与裁剪**: `PerspectiveCamera` / `OrthographicCamera` 提供视图与投影矩阵; `Camera` trait 的默认方法可提取视锥, 与 `Aabb` 组合完成视锥剔除.
- **顶点数据构建**: `VertexFormat` 描述顶点布局; `BufferBuilder` + `Building` 在编译期约束「借出 → 写入 → 结束」的状态机; 写完得到 `RenderedBuffer`, 上传后成为 `GpuMesh`.
- **纹理与绑定资源**: 2D / 3D / Cube 纹理, 采样器, `BindGroupBuilder` / `BindGroupLayoutBuilder`; `MipmapGenerator` 用 compute shader 生成逐级 mipmap.
- **管线状态与缓存**: 用枚举描述混合, 深度, 剔除, 多边形模式, 写掩码等状态, `RenderTypeBuilder` 提供合理默认值; `PipelineCache` 按 `(RenderType, 着色器 id, 布局 id)` 惰性创建并复用 `wgpu` 管线.
- **渲染目标**: 离屏 `RenderTarget` (颜色 + 可选深度 + MSAA), 窗口 `SurfaceFrame`, 独立的 `DepthBuffer` / `MsaaBuffer`; 尺寸变化时通过 `resize` 重建.
- **阴影**: `ShadowMap` 面向方向光 (正交投影), `CubeShadowMap` 面向点光 / 聚光 (立方体深度纹理).
- **后处理**: 线性 `PostProcessChain` 与有向无环图 `PostProcessGraph`, 支持多 pass, 分支与合流; 中间目标池按 ping-pong 策略复用.
- **渲染系统**: `RenderSystem` 持有全局 uniform, 对象 uniform, 共享索引缓冲, 管线缓存等长期资源; `Frame` / `PassHelper` 负责逐 pass 录制, 支持实例化, 蒙皮, 调试图层与 GPU 时间线计时.
- **GPU 读回**: `read_texture` / `read_target` / `read_buffer` 把纹理与缓冲同步复制回 CPU, 用于截图, 测试断言与 compute 结果验证.
- **动画与骨骼**: `Skeleton`, `SkeletonPose`, `AnimationClip`, `BoneTrack` 与 `BoneBuffer` 支撑骨骼动画与蒙皮绘制.

## 模块

| 模块       | 职责                                                           |
|------------|----------------------------------------------------------------|
| `camera`   | 相机抽象, 透视 / 正交相机, 视锥与 AABB 裁剪                    |
| `vertex`   | 顶点格式, 顶点构建器, 索引 / 实例数据, `PoseStack`, 贴花与排序 |
| `texture`  | 纹理, 采样器, bind group 构建, mipmap 生成                     |
| `state`    | 管线状态描述 (混合 / 深度 / 剔除等), 颜色与深度格式, 管线缓存  |
| `shader`   | WGSL 着色器包装与内置着色器集合                                |
| `pipeline` | 渲染目标, 深度 / MSAA 缓冲, 窗口表面, 阴影贴图, 读回           |
| `post`     | 全屏四边形, 后处理链与后处理图                                 |
| `system`   | 长期 GPU 资源, 每帧渲染流程, 网格 / 实例 / 骨骼 / 时间线       |
| `shaders/` | 内置 WGSL 着色器源码 (include_str! 内嵌)                       |

## 渲染流程

一次典型渲染由以下步骤组成:

1. **获取目标** —— 离屏渲染用 `RenderTarget`, 窗口渲染用 `SurfaceFrame::acquire` 从 swap chain 取当前帧纹理.
2. **开始一帧** —— `RenderSystem::begin_frame` 返回 `Frame`, 内部持有 command encoder.
3. **记录 pass** —— `Frame::pass` 开始 render pass, 回调里通过 `PassHelper` 设置管线, 绑定对象, 绘制网格; 计算任务用 `Frame::compute_pass`.
4. **提交** —— `Frame::submit` 结束并提交命令; 启用时间线时同时 resolve GPU timestamp.
5. **呈现** —— 窗口渲染最后调用 `SurfaceFrame::present` 归还图像.

单 pass 场景可以直接用 `RenderSystem::render_frame` 简写.

## 快速开始

```ignore
use std::sync::Arc;

use blaze3d::prelude::*;
use blaze3d::pipeline::{read_target, RenderTarget};

// device / queue 由 wgpu 初始化得到, 这里用 Arc 共享.
let device: Arc<wgpu::Device> = /* ... */;
let queue: Arc<wgpu::Queue> = /* ... */;

// 1. 渲染系统: 长期 GPU 资源.
let mut system = RenderSystem::new(RenderSystemDescriptor {
    device: device.clone(),
    queue: queue.clone(),
    max_vertices: 1 << 20,
    max_objects: 4096,
});

// 2. 离屏渲染目标 (颜色 + 深度).
let target = RenderTarget::new(
    &device,
    1280,
    720,
    ColorFormat::Rgba8Unorm,
    Some(DepthFormat::Depth32Float),
    Some("main_target"),
)?;

// 3. 相机状态写入全局 uniform.
let camera = PerspectiveCamera::new(
    60_f32.to_radians(),
    1280.0 / 720.0,
    0.1,
    1000.0,
);
camera.apply(system.globals_mut());

// 4. 管线描述, 内置着色器与 bind group 布局.
let render_type = RenderType::builder("position_color")
    .vertex_format(formats::position_color())
    .color_format(ColorFormat::Rgba8Unorm)
    .depth_format(Some(DepthFormat::Depth32Float))
    .depth(DepthMode::read_write())
    .cull(CullMode::Back)
    .build();
let shader = BuiltinShader::PositionColor.create(&device);
let layout = LayoutHandle::new(
    &device,
    &[
        system.globals_bind_group_layout(),
        system.object_bind_group_layout(),
    ],
);

// 5. 记录并提交一帧 (mesh 由 BufferBuilder 构建后上传).
system.render_frame(&target, |pass| {
    let offset = pass.push_object(glam::Mat4::IDENTITY).expect("object");
    pass.set_render_type(&render_type, &shader, &layout);
    pass.set_object(offset);
    pass.draw_mesh(&mesh);
});

// 6. 读回像素, 用于截图或测试断言.
let pixels = read_target(&device, &queue, &target)?;
```

内置着色器均附带明确的 bind group 与顶点格式契约, 上例中的 `PositionColor` 使用 group 0 = 全局 uniform, group 1 = 对象 uniform. 示例省略了 `wgpu` 初始化与 `mesh` 构建步骤, 只展示 blaze3d 侧的组织方式.

## 内置着色器

| 变体                                    | 顶点格式                 | group 1   | group 2 | 用途           |
|-----------------------------------------|--------------------------|-----------|---------|----------------|
| `BuiltinShader::PositionColor`          | `position_color`         | `Object`  | —       | 纯顶点色       |
| `BuiltinShader::PositionColorInstanced` | `position_color`         | instances | —       | 实例化绘制     |
| `BuiltinShader::PositionColorSkinned`   | `position_color_skinned` | instances | bones   | 蒙皮           |
| `BuiltinShader::PositionTex`            | `position_tex`           | `Object`  | texture | 纹理与线性雾   |
| `BuiltinShader::PostCopy`               | `position_tex`           | texture   | —       | 后处理直通复制 |

`shaders/` 目录还包含阴影深度, 天空盒, 后处理暗角, mipmap (2D / 3D / Cube) 等 WGSL 着色器. 其中 `PostCopy` 的 group 0 仅保留布局占位, 着色器本身不声明全局 uniform.

## 设计约定

- **bind group 编号**: group 0 恒为全局 uniform; group 1 为对象 uniform, 实例存储或输入纹理; group 2 为骨骼存储或附加纹理. 内置着色器在文档中逐个列出契约.
- **坐标系**: 右手系, 相机看向 `-Z`; NDC 的 `z` 落在 `[0, 1]` (wgpu 约定), 与 OpenGL 的 `[-1, 1]` 不同.
- **管线缓存**: 管线按 `(RenderType, 着色器 id, 布局 id)` 惰性创建; 连续使用同一描述时不会重复绑定.
- **MSAA**: 颜色缓冲与深度缓冲的采样数必须一致; 离屏目标内部管理解析纹理, 窗口渲染由 `SurfaceFrame` 处理 resolve.
- **错误处理**: 所有可能失败的操作返回本模块的 `Error` 枚举 (基于 `thiserror`), 不使用 panic 处理可恢复错误.
- **尺寸约束**: 纹理尺寸会对照设备 `max_texture_dimension_2d` 校验, 越界返回错误而不是静默截断.

## 技术栈

| 依赖        | 版本 | 用途                                           |
|-------------|------|------------------------------------------------|
| `wgpu`      | 30   | 图形 API 抽象 (Vulkan / Metal / DX12 / WebGPU) |
| `glam`      | 0.34 | 向量, 矩阵与四元数                             |
| `bytemuck`  | 1.25 | 顶点数据的 `Pod` 转换                          |
| `thiserror` | 2    | 错误类型派生                                   |

Rust edition 2024, 需要较新的稳定版工具链.

## 构建与文档

```bash
cargo build          # 构建库
cargo test --doc     # 运行文档示例测试
cargo doc --open     # 生成并打开 API 文档
```

生成的文档包含每个模块的职责说明, 关键类型的行为契约, 以及各类型之间的配合关系, 适合作为使用时的第一手参考.
