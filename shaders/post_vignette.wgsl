// 后处理: 采样场景纹理, 应用暗角.
//
// # Bind group 约定
//
// - group 1: 场景纹理 (binding 0 = texture, 1 = sampler).
//
// group 0 的 globals 在本 shader 里不声明 —— 暗角是屏幕空间效果, 不需要
// 相机信息. 但 layout 里可以保留 group 0 占位, 以便与其他后处理着色器
// 共用同一份布局.
//
// # 暗角公式
//
// `vignette = 1 - smoothstep(0.3, 0.75, distance(uv, center)) * 0.7`
//
// 屏幕中心 (uv = (0.5, 0.5)) 处 `distance = 0`, 暗角因子为 1, 不衰减;
// 边缘处 smoothstep 趋于 1, 因子趋于 0.3, 即衰减到原亮度的 30%.

@group(1) @binding(0) var scene_tex: texture_2d<f32>;
@group(1) @binding(1) var scene_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(scene_tex, scene_sampler, in.uv);

    // 暗角: 离屏幕中心越远越暗.
    let d = distance(in.uv, vec2<f32>(0.5, 0.5));
    let vignette = 1.0 - smoothstep(0.3, 0.75, d) * 0.7;

    return vec4<f32>(color.rgb * vignette, color.a);
}
