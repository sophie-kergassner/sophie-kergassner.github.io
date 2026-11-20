// Std of a noise band after upsampling it to full resolution (NoiseGenerator.generate_noise: noise_up.std()).
// Step 1: every 16 x 16 workgroup writes the sum and the sum of squares of its pixels.
// Requires sampling.wgsl.

struct StatsParams {
    factor: f32,         // 2^s
    groups_x: f32,       // number of workgroups per row
    _p0: f32,
    _p1: f32,
}

@group(0) @binding(0) var<uniform> SP: StatsParams;
@group(0) @binding(1) var band: texture_2d<f32>;                         // noise band of level s
@group(0) @binding(2) var full: texture_2d<f32>;                         // any full-resolution texture (for the size)
@group(0) @binding(3) var<storage, read_write> partial: array<vec2<f32>>;

var<workgroup> acc: array<vec2<f32>, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(workgroup_id) wg: vec3<u32>) {
    let dims = textureDimensions(full);
    var v = vec2<f32>(0.0);
    if (id.x < dims.x && id.y < dims.y) {
        let u = upsample_at(band, vec2<i32>(id.xy), SP.factor).r;
        v = vec2<f32>(u, u * u);
    }
    acc[li] = v;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (li < stride) { acc[li] += acc[li + stride]; }
        workgroupBarrier();
    }
    if (li == 0u) { partial[wg.y * u32(SP.groups_x) + wg.x] = acc[0]; }
}
