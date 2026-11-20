// Per frame: eccentricity -> pyramid level -> blur sigma per pixel, and the foveated image as in
// isu.nh_foveate_image: linear interpolation between the two neighbouring layers of the blur stack
// (layer k = image blurred with sigma k; layer 0 = the unblurred image).
// Requires common.wgsl (prepended by the host).

@group(0) @binding(1) var ycc: texture_2d<f32>;                             // unblurred YCbCr (rgba32float)
@group(0) @binding(2) var stack: texture_2d_array<f32>;                     // layer k: blurred with sigma k (rgba16float)
@group(0) @binding(3) var maps: texture_storage_2d<rgba32float, write>;     // (ecc [deg], level, sigma [px], 0)
@group(0) @binding(4) var fov: texture_storage_2d<rgba32float, write>;      // foveated YCbCr

fn stack_layer(p: vec2<i32>, k: i32) -> vec4<f32> {
    if (k <= 0) { return textureLoad(ycc, p, 0); }
    return textureLoad(stack, p, min(k, i32(P.gaze.w)), 0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(fov);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    let rc = vec2<f32>(f32(id.y), f32(id.x));      // (row, col)

    let ecc = eccentricity(rc);
    var level = ecc_to_level(ecc);
    if (P.gaze.z >= 0.0) { level = P.gaze.z; }     // uniform foveation
    let sigma = level_to_sigma(level);

    let k = i32(sigma);                            // truncation, as sigma_map.int()
    let f = sigma - f32(k);
    let v = stack_layer(p, k) * (1.0 - f) + stack_layer(p, k + 1) * f;

    textureStore(maps, p, vec4<f32>(ecc, level, sigma, 0.0));
    textureStore(fov, p, v);
}
