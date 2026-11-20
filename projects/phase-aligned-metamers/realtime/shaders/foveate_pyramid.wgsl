// Per frame: foveation by blending the upsampled Gaussian pyramid levels, as Worm.compute_foveated_imgs.
// The rings (differences of consecutive masks) are nonzero only for the two levels around the pixel's level L,
// so per pixel: fov = (1 - f) * up(level k) + f * up(level k + 1), k = floor(L), f = L - k.
// Requires common.wgsl and sampling.wgsl.

@group(0) @binding(1) var lv0: texture_2d<f32>;
@group(0) @binding(2) var lv1: texture_2d<f32>;
@group(0) @binding(3) var lv2: texture_2d<f32>;
@group(0) @binding(4) var lv3: texture_2d<f32>;
@group(0) @binding(5) var lv4: texture_2d<f32>;
@group(0) @binding(6) var lv5: texture_2d<f32>;
@group(0) @binding(7) var dst: texture_storage_2d<rgba32float, write>;   // foveated YCbCr

fn level_up(i: i32, p: vec2<i32>) -> vec4<f32> {
    switch i {
        case 0: { return textureLoad(lv0, p, 0); }
        case 1: { return upsample_at(lv1, p, 2.0); }
        case 2: { return upsample_at(lv2, p, 4.0); }
        case 3: { return upsample_at(lv3, p, 8.0); }
        case 4: { return upsample_at(lv4, p, 16.0); }
        default: { return upsample_at(lv5, p, 32.0); }
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    var level = ecc_to_level(eccentricity(vec2<f32>(f32(id.y), f32(id.x))));
    if (P.gaze.z >= 0.0) { level = P.gaze.z; }
    let L = clamp(level, 0.0, 5.0);
    let k = min(i32(floor(L)), 4);
    let f = L - f32(k);
    var v = level_up(k, p) * (1.0 - f);
    if (f > 0.0) { v += level_up(k + 1, p) * f; }
    textureStore(dst, p, v);
}
