// Mipmap 生成的 compute shader 模板 (2D).
//
// 采样上一级 Mip, 写入当前级.
//
// # 格式占位符
//
// `{STORAGE_FORMAT}` 在运行时替换为具体格式名 (例如 `rgba8unorm`).
// WGSL 的 storage texture 格式是类型的一部分, 无法参数化 —— 每种格式
// 都需要独立的着色器模块.
//
// # 采样方式
//
// `textureSampleLevel` 是 compute shader 里唯一可用的采样入口,
// `textureSample` 只能在 fragment shader 中使用.
//
// lod 传 0.0: 源视图只包含 1 级 Mip (构造视图时限定), lod 会被 clamp
// 到 0.
//
// # 越界检查
//
// workgroup 尺寸是 8x8, dispatch 时按目标尺寸向上取整, 最后一个
// workgroup 的部分 invocation 会超出目标纹理边界 —— 用
// `textureDimensions` 检查.

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var output_texture: texture_storage_2d<{STORAGE_FORMAT}, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    // 在像素中心采样 —— 双线性在 2x2 上等价于盒式平均.
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims);
    let color = textureSampleLevel(input_texture, input_sampler, uv, 0.0);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
