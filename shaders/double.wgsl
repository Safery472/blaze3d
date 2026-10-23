// 计算管线测试用着色器.
//
// 把 `input` 缓冲的每个元素乘以 2 后写入 `output` 缓冲.
//
// # 越界处理
//
// workgroup 数向上取整 —— 最后一个 workgroup 的 invocation 可能超出
// 数组长度, 用 `arrayLength` 检查.
//
// # 入口点
//
// 只暴露 `cs_main`, 与 render 路径的 `vs_main` / `fs_main` 约定对应.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&input) {
        return;
    }
    output[i] = input[i] * 2.0;
}
