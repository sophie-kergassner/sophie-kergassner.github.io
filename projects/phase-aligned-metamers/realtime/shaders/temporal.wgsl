// Fix 5, temporal phase smoothing (web only): the phase of the statistics of level s, averaged over the last frames.
// Every frame, right after level s is computed (repair_init / repair_level), so the finer levels are extrapolated from
// the smoothed phase. Running average of the phase as a complex number, weighted by the intensity:
//   z = |intensity| exp(i phase),   h_new = h_old + alpha (z - h_old),   phase := angle(h_new)
// alpha = 2 / (N + 1): about the average of the last N frames (N = 3 -> alpha 0.5); alpha = 1 resets (no history).
// N per level (main.js temporalFrames): N_l = 1 + (N - 1) 2^(l - 3), the same lag in pixels of the level.
// theta ~ theta + pi flips the phase sign, so the history is aligned to the current orientation first (as align_sign).
// Intensity, orientation and confidence stay as they are. A static image is unchanged (constant input).
// main:  (rep, history) -> (tmp_rep, tmp_hist);  store: (tmp_rep, tmp_hist) -> (rep, history)
// (two passes, as a texture cannot be read and written in the same pass)

struct TemporalParams {
    alpha: f32,
    _p0: f32, _p1: f32, _p2: f32,
}

@group(0) @binding(0) var<uniform> TP: TemporalParams;
@group(0) @binding(1) var rep_in: texture_2d<f32>;                            // intensity, dir, phase, confidence
@group(0) @binding(2) var hist_in: texture_2d<f32>;                           // (re, im, dir, 0) of the history
@group(0) @binding(3) var rep_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var hist_out: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(rep_out);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    let r = textureLoad(rep_in, p, 0);
    let h = textureLoad(hist_in, p, 0);
    var hz = h.xy;
    if (cos(r.y - h.z) < 0.0) { hz.y = -hz.y; }   // history in the convention of the current orientation
    let z = abs(r.x) * vec2<f32>(cos(r.z), sin(r.z));
    let zs = hz + TP.alpha * (z - hz);
    let phase = select(r.z, atan2(zs.y, zs.x), dot(zs, zs) > 1e-24);
    textureStore(rep_out, p, vec4<f32>(r.x, r.y, phase, r.w));
    textureStore(hist_out, p, vec4<f32>(zs, r.y, 0.0));
}

@group(0) @binding(5) var tmp_rep: texture_2d<f32>;
@group(0) @binding(6) var tmp_hist: texture_2d<f32>;
@group(0) @binding(7) var rep_store: texture_storage_2d<rgba32float, write>;
@group(0) @binding(8) var hist_store: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn store(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(rep_store);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    textureStore(rep_store, p, textureLoad(tmp_rep, p, 0));
    textureStore(hist_store, p, textureLoad(tmp_hist, p, 0));
}
