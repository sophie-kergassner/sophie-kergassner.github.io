// Std of Laplacian band i for the intensity fit, step 1: every 16 x 16 workgroup writes partial sums
// (sum q, sum q x, sum q x^2, pixel count) of its pixels, with
//   weighted fit: q = 1 where the mask of level i is 1 (level <= i: level i fully kept), else 0. Pixels in the soft
//                 transition (mask < 1) are not reliable and are left out. Sampled with nearest neighbour at the
//                 band's size, as Worm.compute_small_masks.
//   cutoff fit:   q = 1 (plain std, as in the paper)

struct LapStatsParams {
    level: f32,          // band i
    groups_x: f32,       // number of workgroups per row
    weighted: f32,       // 1: weighted fit, 0: cutoff fit
    _p0: f32,
}

@group(0) @binding(0) var<uniform> LP: LapStatsParams;
@group(0) @binding(1) var band: texture_2d<f32>;                         // Laplacian band i (r32float)
@group(0) @binding(2) var maps: texture_2d<f32>;                         // full resolution: (ecc, level, sigma, 0)
@group(0) @binding(3) var<storage, read_write> partial: array<vec4<f32>>;

var<workgroup> acc: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(workgroup_id) wg: vec3<u32>) {
    let dims = textureDimensions(band);
    var v = vec4<f32>(0.0);
    if (id.x < dims.x && id.y < dims.y) {
        let p = vec2<i32>(id.xy);
        let x = textureLoad(band, p, 0).r;
        var q = 1.0;
        if (LP.weighted > 0.5) {
            let full = textureDimensions(maps);
            let scale = vec2<f32>(f32(full.x) / f32(dims.x), f32(full.y) / f32(dims.y));
            let fp = min(vec2<i32>(floor(vec2<f32>(p) * scale)), vec2<i32>(full) - 1);
            q = select(0.0, 1.0, textureLoad(maps, fp, 0).y <= LP.level);   // mask == 1
        }
        v = vec4<f32>(q, q * x, q * x * x, 1.0);
    }
    acc[li] = v;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (li < stride) { acc[li] += acc[li + stride]; }
        workgroupBarrier();
    }
    if (li == 0u) { partial[wg.y * u32(LP.groups_x) + wg.x] = acc[0]; }
}
