// Mipmap 生成的 compute shader 模板 (3D 纹理).
//
// 采样上一级 Mip 的体素数据, 写入当前级.
//
// # 格式占位符
//
// `{STORAGE_FORMAT}` 在运行时替换为具体格式名 (例如 `rgba8unorm`).
//
// # 与 2D 版本的差异
//
// - 输出是 `texture_storage_3d`, 而不是 `texture_storage_2d`.
// - 输入是 `texture_3d`, dispatch 是三维的.
// - `textureSampleLevel` 对 3D 纹理使用 `vec3<f32>` 坐标.
//
// # 采样方式
//
// 与 2D 版本相同: 在体素中心采样, 双线性 / 三线性在 2x2x2 上等价于
// 8 抽头平均 (盒式滤波).

@group(0) @binding(0) var input_texture: texture_3d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var output_texture: texture_storage_3d<{STORAGE_FORMAT}, write>;

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (id.x >= dims.x || id.y >= dims.y || id.z >= dims.z) {
        return;
    }
    // 在体素中心采样.
    let uv = (vec3<f32>(id) + 0.5) / vec3<f32>(dims);
    let color = textureSampleLevel(input_texture, input_sampler, uv, 0.0);
    textureStore(output_texture, vec3<i32>(id), color);
}
