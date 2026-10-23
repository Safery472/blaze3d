// Mipmap 生成的 compute shader 模板 (立方体贴图).
//
// 把 cube 当作 6 层的 2D 数组, 每个面独立降采样.
//
// # 格式占位符
//
// `{STORAGE_FORMAT}` 在运行时替换为具体格式名 (例如 `rgba8unorm`).
//
// # 与 2D 版本的差异
//
// - 输入是 `texture_2d_array` (而不是 `texture_cube`), 输出是
//   `texture_storage_2d_array`.
// - dispatch 的 z 维是面索引 (0..6).
// - **不用** `textureSampleLevel`: cube 采样需要方向向量, 会在相邻面
//   共享的棱处产生方向插值, 导致棱两侧的像素来自不同的插值结果. 改用
//   `textureLoad` 按 (坐标, 面索引) 读取, 每个面的 2x2 邻域独立采样,
//   与 CPU 端的盒式滤波一致.
//
// # 绑定编号
//
// 只有 texture 与 storage, **没有采样器**: binding 0 = 输入,
// binding 2 = 输出 —— 与 D2 / D3 版本对齐 (那里 binding 1 是采样器,
// 这里跳过).
//
// # 面索引
//
// `textureStore` 的坐标是 `vec2<i32>` 加上面索引. 面索引由 dispatch
// 的 z 给出 —— 每个面一个 z 层.
//
// # 采样坐标
//
// `textureLoad` 不接受 cube 方向向量. 因此源用 `texture_2d_array`,
// 目标用 `texture_storage_2d_array`, 面索引显式传入.

@group(0) @binding(0) var input_texture: texture_2d_array<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d_array<{STORAGE_FORMAT}, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    // `id.z` 是面索引 (0..6).
    let coord = vec2<i32>(id.xy);
    let layer = i32(id.z);
    let color = textureLoad(input_texture, coord, layer, 0);
    textureStore(output_texture, coord, layer, color);
}
