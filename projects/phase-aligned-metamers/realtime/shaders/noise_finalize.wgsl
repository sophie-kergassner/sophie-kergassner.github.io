// Std of a noise band, step 2: one workgroup adds up all partial sums and writes the (unbiased, as torch.std) std.

struct FinalizeParams {
    n_partials: f32,
    n_pixels: f32,
    scale: f32,          // level s: result goes to stds[s]
    _p0: f32,
}

@group(0) @binding(0) var<uniform> FP: FinalizeParams;
@group(0) @binding(1) var<storage, read> partial: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> stds: array<f32, 8>;

var<workgroup> acc: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    var v = vec2<f32>(0.0);
    for (var i = li; i < u32(FP.n_partials); i += 256u) { v += partial[i]; }
    acc[li] = v;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (li < stride) { acc[li] += acc[li + stride]; }
        workgroupBarrier();
    }
    if (li == 0u) {
        let n = FP.n_pixels;
        let mean = acc[0].x / n;
        stds[u32(FP.scale)] = sqrt(max(acc[0].y - n * mean * mean, 0.0) / (n - 1.0));
    }
}
