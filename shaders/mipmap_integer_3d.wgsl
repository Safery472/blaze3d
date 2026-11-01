// Mipmap 生成的 compute 着色器模板 (3D, 整数格式).
//
// 采样上一级的体素, 写入当前级.
//
// # 格式占位符
//
// - `{STORAGE_FORMAT}` —— 具体格式名 (例如 `rgba8uint`).
// - `{VTYPE}` —— 标量类型, 由 Rust 侧按格式的采样种类填 `u32` 或 `i32`.
//
// # 抽头位置
//
// 输出体素 (x, y, z) 对应源级以 (2x, 2y, 2z) 起始的 2x2x2 块, 八个
// 抽头求和后除以 8. 任一维退化为 1 时该方向只取一个抽头.
//
// 整数纹理不可过滤, 因此不用采样器, Rust 侧的 layout 不声明 binding 1.

@group(0) @binding(0) var input_texture: texture_3d<{VTYPE}>;
@group(0) @binding(2) var output_texture: texture_storage_3d<{STORAGE_FORMAT}, write>;

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (id.x >= dims.x || id.y >= dims.y || id.z >= dims.z) {
        return;
    }

    let src_dims = textureDimensions(input_texture);
    let base = vec3<i32>(id) * 2;
    let stride = vec3<i32>(
        select(1, 2, src_dims.x > 1u),
        select(1, 2, src_dims.y > 1u),
        select(1, 2, src_dims.z > 1u),
    );
    let last = vec3<i32>(src_dims) - 1;
    let zero = vec3<i32>(0);

    // 8 个抽头 = 三维各取 0 或 1 个 stride. 用循环下标 i 的低 3 位
    // 表示这三个分量, 省去三层嵌套循环.
    var sum: vec4<{VTYPE}> = vec4<{VTYPE}>(0);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let bit = vec3<i32>(i32(i & 1u), i32((i >> 1u) & 1u), i32((i >> 2u) & 1u));
        let coord = clamp(base + bit * stride, zero, last);
        sum = sum + textureLoad(input_texture, coord, 0);
    }

    textureStore(output_texture, vec3<i32>(id), sum / {VTYPE}(8));
}
