// 天空盒着色器.
//
// 用全屏四边形绘制, 从 NDC 坐标反投影出世界方向, 再采样 cube 纹理.
//
// # 为什么不用立方体网格
//
// 传统的天空盒使用包围相机的大立方体, 但那样需要:
//
// - 把立方体的 6 个面分别映射到纹理的对应面.
// - 关闭正面剔除, 从内部观察.
// - 保证相机始终位于立方体内部.
//
// 全屏四边形方案更简单: 只需要一次 draw, 方向由 NDC 反投影得到,
// 相机位置不影响结果.
//
// # 方向重建
//
// 对每个屏幕像素, 反投影出 near 平面与 far 平面上的世界点, 二者之差
// 的方向就是该像素对应的世界方向.
//
// # Bind group 约定
//
// - group 0: Globals.
// - group 1: skybox texture (cube) + sampler.
//
// # 顶点格式
//
// 顶点位置必须是 `vec3<f32>` —— 本 shader 用 `.xy` 作为 NDC 并忽略 z.
// 若格式只有 `Float x2`, 应把 `@location(0)` 改为 `vec2<f32>` 并去掉
// `.xy`.

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
@group(1) @binding(0) var skybox_tex: texture_cube<f32>;
@group(1) @binding(1) var skybox_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // position 已经是 NDC. z = 1.0 让它落在远平面, 与场景的深度测试
    // 配合, 场景物体总能覆盖天空盒.
    out.clip_position = vec4<f32>(in.position.xy, 1.0, 1.0);
    out.ndc = in.position.xy;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 从 NDC 反投影出世界空间的两个点.
    //
    // wgpu 的 NDC z 范围是 [0, 1]: near 用 z=0, far 用 z=1.
    // **不要**改成 [-1, 1] —— 那是 OpenGL 的约定.
    let near_h = globals.inverse_view_projection * vec4<f32>(in.ndc, 0.0, 1.0);
    let far_h = globals.inverse_view_projection * vec4<f32>(in.ndc, 1.0, 1.0);

    // 做透视除法, 得到世界坐标.
    let near = near_h.xyz / near_h.w;
    let far = far_h.xyz / far_h.w;

    // near 到 far 的方向就是该像素的视线方向.
    let dir = normalize(far - near);

    return textureSample(skybox_tex, skybox_sampler, dir);
}
