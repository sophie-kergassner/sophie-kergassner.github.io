// Std of Laplacian band i for the intensity fit, step 2: adds up the partial sums of lap_stats.wgsl and writes
// stats[i] = (std, weight):
//   weighted fit: std over the kept region, sqrt(sum q (x - mean)^2 / sum q); weight = kept fraction = mean of q
//   cutoff fit:   unbiased std as torch.std(); weight 1

struct LapFinalizeParams {
    n_partials: f32,
    level: f32,          // band i: result goes to stats[i]
    weighted: f32,
    _p0: f32,
}

@group(0) @binding(0) var<uniform> FP: LapFinalizeParams;
@group(0) @binding(1) var<storage, read> partial: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> stats: array<vec2<f32>, 8>;

var<workgroup> acc: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    var v = vec4<f32>(0.0);
    for (var i = li; i < u32(FP.n_partials); i += 256u) { v += partial[i]; }
    acc[li] = v;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (li < stride) { acc[li] += acc[li + stride]; }
        workgroupBarrier();
    }
    if (li == 0u) {
        let s = acc[0];          // (sum q, sum q x, sum q x^2, count)
        var sd = 0.0;
        var weight = 0.0;
        if (FP.weighted > 0.5) {
            if (s.x >= 1.0) {
                let mean = s.y / s.x;
                sd = sqrt(max(s.z / s.x - mean * mean, 0.0));
                weight = s.x / s.w;
            }
        } else {
            let mean = s.y / s.w;
            sd = sqrt(max(s.z - s.w * mean * mean, 0.0) / (s.w - 1.0));
            weight = 1.0;
        }
        stats[u32(FP.level)] = vec2<f32>(sd, weight);
    }
}
