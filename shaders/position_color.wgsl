// position_color 格式的顶点着色器与片元着色器.
//
// 顶点输入为 position (`@location(0)`) 与 color (`@location(1)`),
// 片元直接输出顶点颜色.
//
// # Bind group 约定
//
// - group 0: Globals uniform.
// - group 1: Object uniform (含 `model` 矩阵).

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

struct Object {
    model: mat4x4<f32>,
}

@group(1) @binding(0)
var<uniform> object: Object;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = object.model * vec4<f32>(input.position, 1.0);
    out.clip_position = globals.view_projection * world;
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
