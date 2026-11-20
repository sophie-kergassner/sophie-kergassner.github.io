// Repair of level s (s < cutoff, computed from coarse to fine), as Worm.repair_intensity / repair_orientation /
// repair_phase. Where the mask of level s is 1, the measured statistics are kept; elsewhere they are extrapolated
// from level s+1 (already repaired). Requires sampling.wgsl, repair_common.wgsl.

@group(0) @binding(1) var maps: texture_2d<f32>;                         // full resolution: (ecc, level, sigma, 0)
@group(0) @binding(2) var ana_s: texture_2d<f32>;                        // analysis of level s: dir, phase, |G + iH|, S
@group(0) @binding(3) var rep_f: texture_2d<f32>;                        // repaired statistics of level s+1
@group(0) @binding(4) var ana_f: texture_2d<f32>;                        // analysis of level s+1
@group(0) @binding(5) var phi: texture_2d<f32>;                          // phi of step 1 (level s+1 size)
@group(0) @binding(6) var rep: texture_storage_2d<rgba32float, write>;   // intensity_rep, dir_rep, phase_rep of level s

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(rep);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    let mask = level_mask(maps, p, dims);
    let a = textureLoad(ana_s, p, 0);
    let src = textureDimensions(rep_f);
    let pos = pos2(p, src);
    let base = floor(pos);
    let t = pos - base;
    let wx = cubic_weights(t.x);
    let wy = cubic_weights(t.y);
    let n = vec2<i32>(src);

    // bicubic upsampling of several fields of level s+1 at once (isu.upsample, factor 2)
    var up_int = 0.0;                    // intensity_rep
    var up_dir = vec2<f32>(0.0);         // exp(2i * dir_rep)   (orientation is defined modulo pi)
    var up_f = vec2<f32>(0.0);           // |G + iH| * exp(i * phase_rep)
    var up_absang = 0.0;                 // |angle(phi)|
    var up_ang = 0.0;                    // angle(phi)
    var up_mag = 0.0;                    // |phi|
    for (var j = 0; j < 4; j++) {
        let y = clamp(i32(base.y) - 1 + j, 0, n.y - 1);
        var r_int = 0.0; var r_dir = vec2<f32>(0.0); var r_f = vec2<f32>(0.0);
        var r_absang = 0.0; var r_ang = 0.0; var r_mag = 0.0;
        for (var i = 0; i < 4; i++) {
            let q = vec2<i32>(clamp(i32(base.x) - 1 + i, 0, n.x - 1), y);
            let rf = textureLoad(rep_f, q, 0);
            let z = textureLoad(phi, q, 0).xy;
            let ang = carg(z);
            r_int += wx[i] * rf.x;
            r_dir += wx[i] * polar(1.0, 2.0 * rf.y);
            r_f += wx[i] * polar(textureLoad(ana_f, q, 0).b, rf.z);
            r_absang += wx[i] * abs(ang);
            r_ang += wx[i] * ang;
            r_mag += wx[i] * length(z);
        }
        up_int += wy[j] * r_int; up_dir += wy[j] * r_dir; up_f += wy[j] * r_f;
        up_absang += wy[j] * r_absang; up_ang += wy[j] * r_ang; up_mag += wy[j] * r_mag;
    }

    // intensity: measured where kept, upsampled coarse intensity elsewhere
    let intensity = a.z * modulator() * mask + up_int * (1.0 - mask);

    // orientation: doubled angle upsampled, halved again, both sides as unit vectors
    let dir_up = polar(length(up_dir), 0.5 * carg(up_dir));
    let fine_dir = polar(1.0, a.x);
    let dir = carg(fine_dir / length(fine_dir) * mask + dir_up / (length(dir_up) + 1e-12) * (1.0 - mask));

    // phase: phu.extrapolate_phi with steps_up = 1
    let phi_up = polar(up_mag, up_absang * sign(up_ang));          // phu.upscale_complex_phase
    let x = 2.0 * carg(up_f) - carg(phi_up) + PI;
    let f_phase = x - 2.0 * PI * floor(x / (2.0 * PI)) - PI;      // (x % 2 pi) - pi, with torch's floor modulo
    let f_recon = polar(length(up_f), f_phase);
    let fine_ph = polar(1.0, a.y);
    let phase = carg(fine_ph / length(fine_ph) * mask + f_recon / (length(f_recon) + 1e-12) * (1.0 - mask));

    textureStore(rep, p, vec4<f32>(intensity, dir, phase, 0.0));
}
