// position_color 的实例化变体.
//
// 通过 `@builtin(instance_index)` 从 group 1 的存储缓冲读取每个实例的
// `model` 矩阵.
//
// # Bind group 约定
//
// - group 0: Globals uniform.
// - group 1: `instances: array<Instance>` (storage, read-only).
//
// # 字段布局
//
// `Instance` 的字段必须与 Rust 侧的 `InstanceData` 严格一致 —— 大小与
// 偏移都要相同, 否则 shader 会按错误的 stride 读取实例数组.
//
// WGSL 里 storage buffer 的 struct 大小等于成员大小之和向上取整到最大
// 对齐. `mat4x4<f32>` 对齐 16, `u32` 对齐 4, 因此 struct 整体对齐 16,
// 大小向上取整到 16 的倍数.
//
// 当前布局:
//   `model`:       offset  0, size 64
//   `bone_offset`: offset 64, size  4
//   `bone_count`:  offset 68, size  4
//   `_pad0`:       offset 72, size  4
//   `_pad1`:       offset 76, size  4
//   总大小: 80 字节 (16 的倍数).
//
// **不要**删除 `bone_offset` / `bone_count` / `_pad*` —— 即使本 shader
// 不用它们. 它们保证 struct 大小是 80 而不是 64, 与 Rust 侧的
// `InstanceData` 一致.

struct Globals {
    view_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    camera_pos: vec4<f32>,
    fog_color: vec4<f32>,
    fog_range: vec2<f32>,
    game_time: f32,
    _pad: f32,
}

@group(0) @binding(0)
var<uniform> globals: Globals;

struct Instance {
    model: mat4x4<f32>,
    bone_offset: u32,
    bone_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(1) @binding(0)
var<storage, read> instances: array<Instance>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(
    input: VertexInput,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let model = instances[instance_index].model;
    let world = model * vec4<f32>(input.position, 1.0);
    out.clip_position = globals.view_projection * world;
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
