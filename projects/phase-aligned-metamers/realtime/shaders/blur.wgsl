// One separable Gaussian blur step of the foveation stack, identical to
// scipy.ndimage.gaussian_filter(x, sigma, truncate=2, mode='reflect') as used in isu.nh_generate_preblurred_image:
//   radius  = int(2 * sigma + 0.5)
//   weights = exp(-0.5 * x^2 / sigma^2), normalized to sum 1
//   border  = 'reflect' (half-sample symmetric: ... c b a | a b c ...)
// scipy filters rows first (vertical), then columns (horizontal); blur_v and blur_h do the same.

struct BlurParams {
    sigma: f32,
    radius: i32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> bp: BlurParams;
@group(0) @binding(1) var src: texture_2d<f32>;                              // rgba32float
@group(0) @binding(2) var dst: texture_storage_2d<rgba32float, write>;       // full-precision result (input of the next step)
@group(0) @binding(3) var dst_layer: texture_storage_2d<rgba16float, write>; // copy for the stack (blur_h only)

fn reflect_idx(i: i32, n: i32) -> i32 {
    var j = i;
    if (j < 0) { j = -j - 1; }
    if (j >= n) { j = 2 * n - j - 1; }
    return j;
}

fn blur(p: vec2<i32>, dir: vec2<i32>, n: i32) -> vec4<f32> {
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var x = -bp.radius; x <= bp.radius; x++) {
        let xf = f32(x);
        let w = exp(-0.5 * xf * xf / (bp.sigma * bp.sigma));
        let q = p + dir * x;
        let qq = select(vec2<i32>(p.x, reflect_idx(q.y, n)), vec2<i32>(reflect_idx(q.x, n), p.y), dir.x != 0);
        acc += w * textureLoad(src, qq, 0);
        wsum += w;
    }
    return acc / wsum;
}

@compute @workgroup_size(8, 8)
fn blur_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    textureStore(dst, p, blur(p, vec2<i32>(0, 1), i32(dims.y)));
}

@compute @workgroup_size(8, 8)
fn blur_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    let v = blur(p, vec2<i32>(1, 0), i32(dims.x));
    textureStore(dst, p, v);
    textureStore(dst_layer, p, v);
}
