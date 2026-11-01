// Mipmap 生成的 compute shader 模板 (2D 数组 / 立方体贴图面, 浮点格式).
//
// 把 cube 当作 6 层的 2D 数组, 每个面独立降采样.
//
// # 格式占位符
//
// - `{STORAGE_FORMAT}` —— 具体格式名 (例如 `rgba8unorm`).
// - `{VTYPE}` —— 标量类型, 由 Rust 侧按格式的采样种类填 `f32`.
//
// # 与 2D 版本的差异
//
// - 输入是 `texture_2d_array` (而不是 `texture_cube`), 输出是
//   `texture_storage_2d_array`.
// - dispatch 的 z 维是面索引 (0..6).
// - **不用** `textureSampleLevel`: cube 采样需要方向向量, 会在相邻面
//   共享的棱处产生方向插值, 导致棱两侧的像素来自不同的插值结果. 改用
//   `textureLoad` 按 (坐标, 面索引) 读取, 每个面的 2x2 邻域独立采样.
//
// # 绑定编号
//
// 只有 texture 与 storage, **没有采样器**: binding 0 = 输入,
// binding 2 = 输出 —— 与 D2 / D3 版本对齐 (那里 binding 1 是采样器,
// 这里跳过).
//
// # 抽头位置
//
// 输出纹素 (x, y) 对应源级以 (2x, 2y) 起始的 2x2 块, 四者取平均.
// 源宽或源高为 1 时该方向只取一个抽头; 四个抽头都夹到源级边界内.
//
// **不要**改成按 (x, y) 原样读取: 那样每个 mip 取到的都是源级的左上角
// 区域, 画面会逐级缩小并偏移, 而不是降采样.

@group(0) @binding(0) var input_texture: texture_2d_array<{VTYPE}>;
@group(0) @binding(2) var output_texture: texture_storage_2d_array<{STORAGE_FORMAT}, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }

    // `id.z` 是面索引 (0..6).
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

    // 除以 4 而不是乘 0.25: 整数格式下 `0.25` 是抽象浮点字面量, 会与
    // `{VTYPE}` 求和结果相加/相乘时要求隐式转换. 显式整数除法与
    // `mipmap_array_integer.wgsl` 的写法一致, 也让 naga IR 直接接受.
    textureStore(output_texture, vec2<i32>(id.xy), layer, sum / {VTYPE}(4));
}
