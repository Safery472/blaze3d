// 蒙皮着色器: position + color + bone_index + bone_weight.
//
// # Bind groups
//
// - group 0: Globals.
// - group 1: instances (storage, read).
// - group 2: bones (storage, read).
//
// # 骨骼矩阵约定
//
// 骨骼矩阵表示**模型空间内的变形**. 蒙皮后的顶点位置为:
//
//     skinned = Σ(weight[i] * bone_matrix[index[i]] * vertex_pos)
//     world   = instance.model * skinned
//
// 与 Minecraft 的骨骼模型一致.
//
// # 无骨骼实例
//
// `instance.bone_count == 0` 时跳过蒙皮, 直接使用 `model` 变换, 因此
// 同一个着色器既能渲染静态实例, 也能渲染蒙皮实例.
//
// # 顶点属性
//
// `@location(2)` 是 `Uint8x4` —— 无符号字节向量, 对应顶点格式里的
// `BoneIndex` 元素.
//
// `@location(3)` 是 `Unorm8x4` —— 归一化到 [0, 1] 的字节向量, 对应
// 顶点格式里的 `BoneWeight`.

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

@group(0) @binding(0) var<uniform> globals: Globals;

struct Instance {
    model: mat4x4<f32>,
    bone_offset: u32,
    bone_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(1) @binding(0) var<storage, read> instances: array<Instance>;
@group(2) @binding(0) var<storage, read> bones: array<mat4x4<f32>>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) bone_indices: vec4<u32>,
    @location(3) bone_weights: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(
    in: VertexInput,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let inst = instances[instance_index];

    // 蒙皮: 4 根骨骼加权混合.
    var skinned = vec4<f32>(in.position, 1.0);
    if inst.bone_count > 0u {
        skinned = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        for (var i = 0u; i < 4u; i = i + 1u) {
            let w = in.bone_weights[i];
            if w > 0.0 {
                let idx = inst.bone_offset + in.bone_indices[i];
                let m = bones[idx];
                skinned += w * (m * vec4<f32>(in.position, 1.0));
            }
        }
    }

    let world = inst.model * skinned;

    var out: VertexOutput;
    out.clip_position = globals.view_projection * world;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
