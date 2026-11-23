// Orientation stability fix (web only, debug toggle #orientBtn): C2, C3 averaged over a Gaussian window
// (7x7, sigma ORIENT_SIGMA px of the level, zero padding) instead of the 5x5 box of phu.get_theta_d, as two separable passes.
// dir = atan2(C3, C2) / 2 is undefined where C ~ 0; there, small changes of the band (e.g. the image moving by a pixel)
// turn it by large angles, and G, H and the phase are steered with it. The wider window averages out these small-scale
// fluctuations. Headless measurement (1-px shifts, face.jpg, level 0): magnitude-weighted mean frame-to-frame change
// of dir 6.7 deg (box) -> 5.3 deg (7x7, sigma 2); share of jumps > 30 deg 3.6 % -> 2.3 %. A 17x17 window (sigma 3)
// gave 2.9 deg / 0.7 %, but smoothed the orientation map too much; a 5x5 Gaussian (sigma 1) is worse than the box.
// smooth_h: cc (C2, C3) -> tmp;  smooth_v: tmp -> cc.  Then analysis_orient.wgsl main_presmoothed.
// smooth_h2, smooth_v2 (Fix 4, dense coarse evaluation): taps 2 px apart, the window of level l+1 at level l.

const ORIENT_SIGMA: f32 = 2.0;
const ORIENT_RADIUS: i32 = 3;   // 7x7 window (truncated at 1.5 sigma)

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;

fn window_avg(p: vec2<i32>, step: vec2<i32>) -> vec2<f32> {
    let n = vec2<i32>(textureDimensions(src));
    var acc = vec2<f32>(0.0);
    var wsum = 0.0;
    for (var k = -ORIENT_RADIUS; k <= ORIENT_RADIUS; k++) {
        let w = exp(-0.5 * f32(k * k) / (ORIENT_SIGMA * ORIENT_SIGMA));
        wsum += w;
        let q = p + step * k;
        if (q.x >= 0 && q.y >= 0 && q.x < n.x && q.y < n.y) {   // zero padding (as the box filter)
            acc += w * textureLoad(src, q, 0).rg;
        }
    }
    return acc / wsum;
}

@compute @workgroup_size(8, 8)
fn smooth_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    textureStore(dst, p, vec4<f32>(window_avg(p, vec2<i32>(1, 0)), 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn smooth_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    textureStore(dst, p, vec4<f32>(window_avg(p, vec2<i32>(0, 1)), 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn smooth_h2(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    textureStore(dst, p, vec4<f32>(window_avg(p, vec2<i32>(2, 0)), 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn smooth_v2(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    textureStore(dst, p, vec4<f32>(window_avg(p, vec2<i32>(0, 2)), 0.0, 0.0));
}
