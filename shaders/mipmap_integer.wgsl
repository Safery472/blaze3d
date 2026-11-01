// Mipmap 生成的 compute shader 模板 (2D, 整数格式).
//
// 采样上一级 Mip, 写入当前级.
//
// # 格式占位符
//
// - `{STORAGE_FORMAT}` —— 具体格式名 (例如 `rgba8uint`).
// - `{VTYPE}` —— 标量类型, 由 Rust 侧按格式的采样种类填 `u32` 或 `i32`.
//
// 整数 storage 纹理只能接受对应的整数向量: `rgba8uint` 必须用
// `vec4<u32>` 写入, `vec4<f32>` 无法隐式转换. 因此本模板与
// `mipmap.wgsl` 分开.
//
// # 为什么不用采样器
//
// 整数纹理不可过滤, 也没有双线性的概念 —— 采样器在本路径上无用, Rust
// 侧的 layout 因此不声明 binding 1.
//
// # 抽头位置
//
// 输出纹素 (x, y) 对应源级以 (2x, 2y) 起始的 2x2 块, 四个抽头求和后
// 除以 4. 整数除法向下取整, 结果即块平均值向下取整.
//
// 源级宽高未必恰好是目标级的两倍: 源宽或源高为 1 时该方向无法配对,
// 只取一个抽头. 四个抽头都夹到源级边界内, 源级为奇数尺寸时最后一个块
// 不会越界.

@group(0) @binding(0) var input_texture: texture_2d<{VTYPE}>;
@group(0) @binding(2) var output_texture: texture_storage_2d<{STORAGE_FORMAT}, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }

    let src_dims = textureDimensions(input_texture);
    let base = vec2<i32>(id.xy) * 2;
    let stride = vec2<i32>(select(1, 2, src_dims.x > 1u), select(1, 2, src_dims.y > 1u));
    let last = vec2<i32>(src_dims) - 1;
    let zero = vec2<i32>(0);

    let c00 = clamp(base, zero, last);
    let c10 = clamp(base + vec2<i32>(stride.x, 0), zero, last);
    let c01 = clamp(base + vec2<i32>(0, stride.y), zero, last);
    let c11 = clamp(base + stride, zero, last);

    let sum = textureLoad(input_texture, c00, 0)
        + textureLoad(input_texture, c10, 0)
        + textureLoad(input_texture, c01, 0)
        + textureLoad(input_texture, c11, 0);

    textureStore(output_texture, vec2<i32>(id.xy), sum / {VTYPE}(4));
}
