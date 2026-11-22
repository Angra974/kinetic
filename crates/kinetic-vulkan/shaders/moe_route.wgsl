// The router of a mixture of experts: softmax over the experts' logits,
// then the `k` likeliest experts of each token, weighted by their share of
// those `k` probabilities and written as the slots the grouped GEMV runs.
//
// One invocation routes one token: the router is a few hundred logits
// against the width of the block, and the walk over the experts would cost
// more in a reduction than it saves.

struct Shape {
    experts: u32,
    k: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
// The slot count, then the expert each slot runs.
@group(0) @binding(2) var<storage, read_write> moe: array<u32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;

const MAX_EXPERTS: u32 = 1024u;
const MAX_PICKS: u32 = 32u;
/// The smallest normal f16, the floor the reference puts under the sum.
const FLOOR: f32 = 6.1035156e-5;

var<workgroup> probs: array<f32, 1024>;
var<workgroup> picked: array<u32, 32>;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) group: vec3<u32>) {
    let row = group.x;
    if row >= batch[0] {
        return;
    }
    let n = min(shape.experts, MAX_EXPERTS);
    let k = min(shape.k, MAX_PICKS);
    let at = row * n;
    var best = logits[at];
    for (var e = 1u; e < n; e++) {
        best = max(best, logits[at + e]);
    }
    var total = 0.0;
    for (var e = 0u; e < n; e++) {
        let p = exp(logits[at + e] - best);
        probs[e] = p;
        total += p;
    }
    for (var e = 0u; e < n; e++) {
        probs[e] = probs[e] / total;
    }

    // The k likeliest, largest first; an equal probability keeps the lower
    // expert, as the reference's sort does.
    var share = 0.0;
    for (var j = 0u; j < k; j++) {
        var pick = 0u;
        var top = -1.0;
        for (var e = 0u; e < n; e++) {
            var taken = false;
            for (var t = 0u; t < j; t++) {
                taken = taken || picked[t] == e;
            }
            if (!taken && probs[e] > top) {
                top = probs[e];
                pick = e;
            }
        }
        picked[j] = pick;
        share += top;
    }
    share = max(share, FLOOR);
    for (var j = 0u; j < k; j++) {
        let slot = row * k + j;
        weights[slot] = probs[picked[j]] / share;
        moe[1u + slot] = picked[j];
    }
    moe[0] = batch[0] * k;
}
