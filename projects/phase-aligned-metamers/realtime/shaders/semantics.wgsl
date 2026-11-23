// Phase semantics of level l (display only, "Line / Edge" view): the phase left after matching the coarse level's
// frequency to the fine one, as phi in phu.get_phi_quick, but with level l as the fine and level l+1 as the coarse one:
//   f     = |G + iH|_l     * exp(i * phase_l)
//   c_up  = isu.upsample(|G + iH|_{l+1} * exp(i * phase_{l+1}))   (bicubic, complex; as up_f in repair_level.wgsl)
//   sem   = angle(c_up^2 / |c_up| * conj(f)) = 2 phase_{l+1} - phase_l
// At a line or edge the phase is the same at all scales, so sem = that phase (0 / pi: line, +-pi/2: edge).
// Where level l is extrapolated, phase_l = 2 angle(c_up) - angle(phi_up) (repair_level.wgsl), so sem = angle(phi_up):
// the semantics of level l+1, only upsampled; the semantics map stays the same down the extrapolated levels.
// RP.scale = l; RP._p0 = 1: measured phases (analysis .y), 0: repaired phases (repaired .z).
// Output: (sem, strength, 0, 0), strength = |G + iH|_l (measured) or intensity_rep_l (repaired), as the intensity views.
// main_dense (Fix 4, dense coarse evaluation): c_up is not interpolated from the pixels of level
// l+1 but taken from the filters of level l+1 evaluated at every pixel of level l (ana_cd: pyramid_down.wgsl blur_dense,
// analysis_filters.wgsl main_dilated, orientation with taps 2 px apart). At the pixels of level l+1 it is the same value;
// in between, the bicubic interpolation of the fast-rotating coarse phase depended on where the image sits relative to
// the coarse grid, so a moving image changed the Line / Edge judgement of whole regions. The pyramid is unchanged.
// Repaired phases: where level l+1 is extrapolated (below the cutoff, outside its mask) its phase exists only on its own
// grid and stays interpolated; c = mask_{l+1} * dense + (1 - mask_{l+1}) * interpolated (unit vectors, as repair_level).
// RP.orient (main_dense only) = 1: level l+1 is measured everywhere (measured variant, or level l+1 >= cutoff).
// Requires sampling.wgsl, repair_common.wgsl.

@group(0) @binding(1) var ana_f: texture_2d<f32>;                        // analysis of level l: dir, phase, |G + iH|, S
@group(0) @binding(2) var rep_f: texture_2d<f32>;                        // repaired statistics of level l
@group(0) @binding(3) var ana_c: texture_2d<f32>;                        // analysis of level l+1
@group(0) @binding(4) var rep_c: texture_2d<f32>;                        // repaired statistics of level l+1
@group(0) @binding(5) var sem: texture_storage_2d<rgba32float, write>;   // (sem, strength, 0, 0) at the size of level l
@group(0) @binding(6) var ana_cd: texture_2d<f32>;                       // main_dense: analysis of level l+1 at every pixel of level l
@group(0) @binding(7) var maps: texture_2d<f32>;                         // main_dense: full resolution (ecc, level, sigma, 0)

fn phase_at(ana: texture_2d<f32>, rep: texture_2d<f32>, q: vec2<i32>) -> f32 {
    return select(textureLoad(rep, q, 0).z, textureLoad(ana, q, 0).y, RP._p0 > 0.5);
}
fn dir_at(ana: texture_2d<f32>, rep: texture_2d<f32>, q: vec2<i32>) -> f32 {
    return select(textureLoad(rep, q, 0).y, textureLoad(ana, q, 0).x, RP._p0 > 0.5);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(sem);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    store_sem(p, interp_c(p, dir_at(ana_f, rep_f, p)));
}

// bicubic upsampling of the complex field of level l+1 (bicubic_mag_phase, with the phase source selectable),
// each tap aligned to the orientation of level l at p (align_sign)
fn interp_c(p: vec2<i32>, theta_ref: f32) -> vec2<f32> {
    let src = textureDimensions(ana_c);
    let pos = pos2(p, src);
    let base = floor(pos);
    let t = pos - base;
    let wx = cubic_weights(t.x);
    let wy = cubic_weights(t.y);
    let n = vec2<i32>(src);
    var c_up = vec2<f32>(0.0);
    for (var j = 0; j < 4; j++) {
        let y = clamp(i32(base.y) - 1 + j, 0, n.y - 1);
        var row = vec2<f32>(0.0);
        for (var i = 0; i < 4; i++) {
            let q = vec2<i32>(clamp(i32(base.x) - 1 + i, 0, n.x - 1), y);
            row += wx[i] * polar(textureLoad(ana_c, q, 0).b, align_sign(dir_at(ana_c, rep_c, q), theta_ref) * phase_at(ana_c, rep_c, q));
        }
        c_up += wy[j] * row;
    }
    return c_up;
}

fn store_sem(p: vec2<i32>, c_up: vec2<f32>) {
    let a = textureLoad(ana_f, p, 0);
    let f = polar(1.0, phase_at(ana_f, rep_f, p));   // only its angle matters (|G + iH|_l is ~0 where blurred)
    let c_hat = cmul(c_up, c_up) / (length(c_up) + 1e-6);
    let strength = select(textureLoad(rep_f, p, 0).x, a.z, RP._p0 > 0.5);
    textureStore(sem, p, vec4<f32>(carg(cmul(c_hat, conj(f))), strength, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn main_dense(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(sem);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    let theta_ref = dir_at(ana_f, rep_f, p);
    let d = textureLoad(ana_cd, p, 0);   // dir, phase, |G + iH|, S of level l+1 at p (only the angle of c is used)
    var c = polar(1.0, align_sign(d.x, theta_ref) * d.y);
    if (RP.orient < 0.5) {   // level l+1 partly extrapolated: blend with the interpolated repaired phase by its mask
        let full = textureDimensions(maps);
        let scale = vec2<f32>(f32(full.x) / f32(dims.x), f32(full.y) / f32(dims.y));
        let q = min(vec2<i32>(floor(vec2<f32>(p) * scale)), vec2<i32>(full) - 1);
        let m = 1.0 - clamp(textureLoad(maps, q, 0).y - (RP.scale + 1.0), 0.0, 1.0);
        let ci = interp_c(p, theta_ref);
        c = c * m + ci / (length(ci) + 1e-12) * (1.0 - m);
    }
    store_sem(p, c);
}
