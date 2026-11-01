// 阴影深度 pass 的着色器 (普通网格版本).
//
// 只需要顶点位置 —— 深度由固定的深度附件写入.
//
// # 与 `shadow_depth.wgsl` 的差异
//
// 那份是**实例化**版本, 从 group 1 的 storage buffer 按
// `instance_index` 取 model 矩阵, 只能配合 `draw_mesh_instanced`.
// 本份读 group 1 的 Object uniform, 配合 `PassHelper::draw_mesh` 与
// `PassHelper::set_object` 使用.
//
// # Bind group 约定
//
// - group 0: Globals uniform (`view_projection` 已替换为光源视角的矩阵).
// - group 1: Object uniform (`model`), 动态偏移选择对象.
//
// # 片元阶段
//
// 片元返回一个常量颜色, 但**调用方必须**在管线里关闭颜色写入
// (`WriteMask::DEPTH_ONLY`), 否则会覆盖颜色附件. 阴影 pass 只需要深度,
// 颜色附件的 dummy 值会被丢弃.
//
// wgpu 要求 `color_targets` 非空的管线必须带片元阶段, 因此这里无法
// 省略.

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

struct Object {
    model: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> globals: Globals;

@group(1) @binding(0)
var<uniform> object: Object;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> @builtin(position) vec4<f32> {
    let world = object.model * vec4<f32>(input.position, 1.0);
    return globals.view_projection * world;
}

// 片元不输出任何有效颜色 —— 深度由附件写入. 但 wgpu 要求 color_targets
// 非空的管线必须带片元阶段, 因此这里返回常量颜色, 实际写入由管线的
// write_mask 关闭.
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
