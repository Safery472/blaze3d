// Mipmap 生成的 compute shader 模板 (2D 数组 / 立方体贴图面, 整数格式).
//
// 与 `mipmap_array.wgsl` 的差别只在向量类型: 整数 storage 纹理只能接受
// `vec4<u32>` / `vec4<i32>`, 因此除法也按整数进行 (向下取整).
//
// # 格式占位符
//
// - `{STORAGE_FORMAT}` —— 具体格式名 (例如 `rgba8uint`).
// - `{VTYPE}` —— 标量类型, 由 Rust 侧按格式的采样种类填 `u32` 或 `i32`.
//
// # 抽头位置
//
// 输出纹素 (x, y) 对应源级以 (2x, 2y) 起始的 2x2 块, 四个抽头求和后
// 整除 4. 没有采样器, binding 0 是输入, binding 2 是输出.

@group(0) @binding(0) var input_texture: texture_2d_array<{VTYPE}>;
@group(0) @binding(2) var output_texture: texture_storage_2d_array<{STORAGE_FORMAT}, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }

    let layer = i32(id.z);
    let src_dims = textureDimensions(input_texture).xy;

    let base = vec2<i32>(id.xy) * 2;
    let stride = vec2<i32>(select(1, 2, src_dims.x > 1u), select(1, 2, src_dims.y > 1u));
    let last = vec2<i32>(src_dims) - 1;
    let zero = vec2<i32>(0);

    let c00 = clamp(base, zero, last);
    let c10 = clamp(base + vec2<i32>(stride.x, 0), zero, last);
    let c01 = clamp(base + vec2<i32>(0, stride.y), zero, last);
    let c11 = clamp(base + stride, zero, last);

    let sum = textureLoad(input_texture, c00, layer, 0)
        + textureLoad(input_texture, c10, layer, 0)
        + textureLoad(input_texture, c01, layer, 0)
        + textureLoad(input_texture, c11, layer, 0);

    textureStore(output_texture, vec2<i32>(id.xy), layer, sum / {VTYPE}(4));
}
