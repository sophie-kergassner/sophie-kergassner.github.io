// Phase repair of level s, step 1 (computed at the resolution of level s+1), phu.get_phi_quick:
//   f     = |G + iH|_{s+1} * exp(i * phase_rep_{s+1})
//   c     = |G + iH|_{s+2} * exp(i * phase_rep_{s+2}),  c_up = isu.upsample(c)  (to the size of level s+1)
//   c_hat = c_up^2 / (|c_up| + 1e-6)
//   phi   = c_hat * conj(f)
// Requires sampling.wgsl, repair_common.wgsl.

@group(0) @binding(1) var ana_f: texture_2d<f32>;                        // analysis of level s+1
@group(0) @binding(2) var rep_f: texture_2d<f32>;                        // repaired statistics of level s+1
@group(0) @binding(3) var ana_c: texture_2d<f32>;                        // analysis of level s+2
@group(0) @binding(4) var rep_c: texture_2d<f32>;                        // repaired statistics of level s+2
@group(0) @binding(5) var phi: texture_storage_2d<rgba32float, write>;   // (re, im, 0, 0) at the size of level s+1

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(phi);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    let c_up = bicubic_mag_phase(ana_c, rep_c, pos2(p, textureDimensions(ana_c)));
    let f = polar(textureLoad(ana_f, p, 0).b, textureLoad(rep_f, p, 0).z);
    let c_hat = cmul(c_up, c_up) / (length(c_up) + 1e-6);
    textureStore(phi, p, vec4<f32>(cmul(c_hat, conj(f)), 0.0, 0.0));
}
