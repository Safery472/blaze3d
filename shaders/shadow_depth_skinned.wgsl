// 阴影深度 pass 的着色器 (蒙皮网格版本).
//
// 只需要顶点位置与骨骼属性 —— 深度由固定的深度附件写入.
//
// # 与另外两份的差异
//
// - `shadow_depth.wgsl`: 实例化, group 1 是 storage buffer 里的实例数组.
// - `shadow_depth_mesh.wgsl`: 普通网格, group 1 是 Object uniform.
// - 本份: 普通网格 + 蒙皮. model 与骨骼范围都来自 Object uniform, 骨骼
//   矩阵来自 group 2.
//
// 蒙皮模型的阴影必须跟随骨骼变形, 否则主 pass 里变形后的顶点与阴影
// pass 里的原始顶点会对不上.
//
// # 骨骼矩阵约定
//
// 与 `position_color_skinned.wgsl` 一致:
//
//     skinned = Σ(weight[i] * bone_matrix[bone_start + index[i]] * pos)
//     world   = object.model * skinned
//
// `bone_count == 0` 时跳过蒙皮, 直接使用 `model` 变换.
//
// # Bind group 约定
//
// - group 0: Globals uniform (`view_projection` 已替换为光源视角的矩阵).
// - group 1: Object uniform (`model` + `bone_start` + `bone_count`).
// - group 2: bones (`array<mat4x4<f32>>`, storage read).
//
// # 顶点属性
//
// - `@location(0)` position.
// - `@location(2)` `Uint8x4` 骨骼索引.
// - `@location(3)` `Unorm8x4` 骨骼权重.

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
    bone_start: u32,
    bone_count: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0)
var<uniform> globals: Globals;

@group(1) @binding(0)
var<uniform> object: Object;

@group(2) @binding(0)
var<storage, read> bones: array<mat4x4<f32>>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(2) bone_indices: vec4<u32>,
    @location(3) bone_weights: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> @builtin(position) vec4<f32> {
    var skinned = vec4<f32>(input.position, 1.0);
    if object.bone_count > 0u {
        skinned = vec4<f32>(0.0);
        for (var i = 0u; i < 4u; i = i + 1u) {
            let w = input.bone_weights[i];
            if w > 0.0 {
                let idx = object.bone_start + input.bone_indices[i];
                skinned += w * (bones[idx] * vec4<f32>(input.position, 1.0));
            }
        }
    }

    let world = object.model * skinned;
    return globals.view_projection * world;
}

// 片元返回常量颜色; 实际写入由管线的 write_mask 关闭.
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
