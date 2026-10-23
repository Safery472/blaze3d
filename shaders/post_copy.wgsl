// 后处理: 直接把输入纹理复制到输出.
//
// 不做任何效果 —— 用于「把场景原样送到窗口」, 或作为后处理链的占位
// pass.
//
// # Bind group 约定
//
// - group 1: 输入纹理 (binding 0 = texture, 1 = sampler).
//
// group 0 的 globals 在本 shader 里不声明 —— 后处理不需要帧级 uniform.
// 但 layout 里可以保留 group 0, 以便与其他后处理着色器共用同一份布局.
//
// # 顶点格式
//
// position_tex —— 全屏四边形. Position 已经是 NDC, 直接输出.

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(1) @binding(0)
var input_texture: texture_2d<f32>;

@group(1) @binding(1)
var input_sampler: sampler;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // 全屏四边形的 Position 已经是 NDC, 直接输出.
    out.clip_position = vec4<f32>(input.position, 1.0);
    out.uv = input.uv;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(input_texture, input_sampler, input.uv);
}
