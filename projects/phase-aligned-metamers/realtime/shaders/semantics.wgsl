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
// Requires sampling.wgsl, repair_common.wgsl.

@group(0) @binding(1) var ana_f: texture_2d<f32>;                        // analysis of level l: dir, phase, |G + iH|, S
@group(0) @binding(2) var rep_f: texture_2d<f32>;                        // repaired statistics of level l
@group(0) @binding(3) var ana_c: texture_2d<f32>;                        // analysis of level l+1
@group(0) @binding(4) var rep_c: texture_2d<f32>;                        // repaired statistics of level l+1
@group(0) @binding(5) var sem: texture_storage_2d<rgba32float, write>;   // (sem, strength, 0, 0) at the size of level l

fn phase_at(ana: texture_2d<f32>, rep: texture_2d<f32>, q: vec2<i32>) -> f32 {
    return select(textureLoad(rep, q, 0).z, textureLoad(ana, q, 0).y, RP._p0 > 0.5);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(sem);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);

    // bicubic upsampling of the complex field of level l+1 (bicubic_mag_phase, with the phase source selectable)
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
            row += wx[i] * polar(textureLoad(ana_c, q, 0).b, phase_at(ana_c, rep_c, q));
        }
        c_up += wy[j] * row;
    }

    let a = textureLoad(ana_f, p, 0);
    let f = polar(1.0, phase_at(ana_f, rep_f, p));   // only its angle matters (|G + iH|_l is ~0 where blurred)
    let c_hat = cmul(c_up, c_up) / (length(c_up) + 1e-6);
    let strength = select(textureLoad(rep_f, p, 0).x, a.z, RP._p0 > 0.5);
    textureStore(sem, p, vec4<f32>(carg(cmul(c_hat, conj(f))), strength, 0.0, 0.0));
}
