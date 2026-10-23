// position_tex 格式: position + uv.
//
// 采样纹理并叠加线性雾. 这是最常用的「带纹理」着色器, 后续内置着色器
// (光照, 雾) 都在它的基础上扩展.
//
// # Bind group 约定
//
// - group 0: Globals uniform.
// - group 1: Object uniform (`model` 矩阵).
// - group 2: 基础颜色纹理 (binding 0 = texture, 1 = sampler).
//
// # 雾效
//
// 线性雾: 相机距离 `<= fog_range.x` 时因子为 0, `>= fog_range.y` 时为
// 1, 中间线性插值. `fog_range.y == fog_range.x` 时结果未定义 —— 调用方
// 需保证二者不相等.

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

@group(2) @binding(0)
var base_texture: texture_2d<f32>;

@group(2) @binding(1)
var base_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    // 世界空间坐标 —— 供雾效计算使用.
    @location(1) world_position: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = object.model * vec4<f32>(input.position, 1.0);
    out.clip_position = globals.view_projection * world;
    out.uv = input.uv;
    out.world_position = world.xyz;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(base_texture, base_sampler, input.uv);

    // 线性雾: 相机距离超过 fog_range.y 后完全为雾色.
    let to_camera = globals.camera_pos.xyz - input.world_position;
    let distance = length(to_camera);
    let fog_factor = clamp(
        (distance - globals.fog_range.x) / (globals.fog_range.y - globals.fog_range.x),
        0.0,
        1.0,
    );
    color = vec4<f32>(
        mix(color.rgb, globals.fog_color.rgb, fog_factor),
        color.a,
    );

    return color;
}
