// Analysis, step 1 (per pyramid level, Y channel): the 7 steerable basis responses
// R = [Ga, Gb, Gc, Ha, Hb, Hc, Hd] (phu.get_basis_images: 9x9 filters, torch 'reflect' padding, correlation)
// and the orientation energy terms C2, C3 (phu.get_C2 / get_C3, before blurring).
// main_dilated (Fix 4, dense coarse evaluation): the 9x9 filters with their taps 2 px apart, on the blurred level l at
// full size (pyramid_down.wgsl blur_dense): the filters of level l+1, at the same size, evaluated at every pixel of level l.
// Requires sampling.wgsl.

@group(0) @binding(0) var<storage, read> filt: array<f32, 567>;          // 7 filters x 81 weights (constants.json)
@group(0) @binding(1) var lvl: texture_2d<f32>;                          // pyramid level (rgba32float, Y in .r)
@group(0) @binding(2) var ra: texture_storage_2d<rgba32float, write>;    // Ga, Gb, Gc, Ha
@group(0) @binding(3) var rb: texture_storage_2d<rgba32float, write>;    // Hb, Hc, Hd, 0
@group(0) @binding(4) var cc: texture_storage_2d<rgba32float, write>;    // C2, C3, 0, 0

fn basis(p: vec2<i32>, n: vec2<i32>, dil: i32) {

    var R: array<f32, 7>;
    for (var f = 0; f < 7; f++) { R[f] = 0.0; }
    for (var ky = 0; ky < 9; ky++) {
        for (var kx = 0; kx < 9; kx++) {
            let q = vec2<i32>(reflect_torch(p.x + dil * (kx - 4), n.x), reflect_torch(p.y + dil * (ky - 4), n.y));
            let v = textureLoad(lvl, q, 0).r;
            for (var f = 0; f < 7; f++) {
                R[f] += filt[f * 81 + ky * 9 + kx] * v;
            }
        }
    }
    let Ga = R[0]; let Gb = R[1]; let Gc = R[2];
    let Ha = R[3]; let Hb = R[4]; let Hc = R[5]; let Hd = R[6];

    let C2 = 0.5 * (Gb * Gb - Gc * Gc) + 0.46875 * (Ha * Ha - Hd * Hd)
           + 0.28125 * (Hb * Hb - Hc * Hc) + 0.1875 * (Ha * Hc - Hb * Hd);
    let C3 = -Ga * Gb - Gb * Gc
           - 0.9375 * (Hc * Hd + Ha * Hb) - 1.6875 * Hb * Hc - 0.1875 * Ha * Hd;

    textureStore(ra, p, vec4<f32>(Ga, Gb, Gc, Ha));
    textureStore(rb, p, vec4<f32>(Hb, Hc, Hd, 0.0));
    textureStore(cc, p, vec4<f32>(C2, C3, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(ra);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    basis(vec2<i32>(id.xy), vec2<i32>(dims), 1);
}

@compute @workgroup_size(8, 8)
fn main_dilated(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(ra);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    basis(vec2<i32>(id.xy), vec2<i32>(dims), 2);
}
