// Mixes the routed experts' outputs into the block's output: row `t` of
// `y` is the sum over the token's `k` slots of the slot's weight times its
// row of `out`. The slot count is the first word of `moe`.

struct Shape {
    d: u32,
    k: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> moe: array<u32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> out: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let i = group.x * 256u + tid;
    let row = group.y;
    if i >= shape.d || row >= moe[0] / shape.k {
        return;
    }
    var sum = 0.0;
    for (var j = 0u; j < shape.k; j++) {
        let slot = row * shape.k + j;
        sum += weights[slot] * out[slot * shape.d + i];
    }
    y[row * shape.d + i] = sum;
}
