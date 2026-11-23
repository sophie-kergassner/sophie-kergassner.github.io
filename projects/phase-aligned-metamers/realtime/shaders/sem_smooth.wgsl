// Line / Edge view, display form (display only). The phase semantics sem of level l (semantics.wgsl)
// is shown as edge-ness e = sin^2(sem) = (1 - cos(2 sem)) / 2 (0 line, 1 edge), faded by the strength. In textures
// sem varies from pixel to pixel, and small sub-pixel changes of the coarse level recolour it from frame to frame.
// Here the map is only upsampled smoothly to the screen (display.wgsl, bicubic instead of nearest blocks). So that the
// interpolation acts on e (and not on the angle, which wraps), this pass stores
//   out = (strength * cos(2 sem), strength)   ->   e = (1 - x / y) / 2,  strength = y   (strength-weighted e)
// SEM_RADIUS > 0 would also average over a window, but in pixels of level l, i.e. 2^l times wider on screen at coarse
// levels than at fine ones: a 7x7 window made L3 look blurred and L0 sharp. Hence no averaging (radius 0).
// sem_h: sem (sem, strength) -> tmp;  sem_v: tmp -> out.

const SEM_SIGMA: f32 = 2.0;
const SEM_RADIUS: i32 = 0;   // no averaging, only the conversion for the smooth upsampling

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;

fn z_at(q: vec2<i32>, from_sem: bool) -> vec2<f32> {
    let v = textureLoad(src, q, 0);
    if (from_sem) { return vec2<f32>(v.y * cos(2.0 * v.x), v.y); }
    return v.xy;
}

// Gaussian average along `step` over the taps inside the texture (weights renormalised at the border)
fn window_avg(p: vec2<i32>, step: vec2<i32>, from_sem: bool) -> vec2<f32> {
    let n = vec2<i32>(textureDimensions(src));
    var acc = vec2<f32>(0.0);
    var wsum = 0.0;
    for (var k = -SEM_RADIUS; k <= SEM_RADIUS; k++) {
        let q = p + step * k;
        if (q.x >= 0 && q.y >= 0 && q.x < n.x && q.y < n.y) {
            let w = exp(-0.5 * f32(k * k) / (SEM_SIGMA * SEM_SIGMA));
            acc += w * z_at(q, from_sem);
            wsum += w;
        }
    }
    return acc / wsum;
}

@compute @workgroup_size(8, 8)
fn sem_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    textureStore(dst, p, vec4<f32>(window_avg(p, vec2<i32>(1, 0), true), 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn sem_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    textureStore(dst, p, vec4<f32>(window_avg(p, vec2<i32>(0, 1), false), 0.0, 0.0));
}

// Sureness of the Line / Edge judgement (display only, Phase view "Sureness"). The judgement of level l flickers when
// the image moves because the level samples the image at a different sub-pixel position every frame: where line / edge
// changes from one level-l pixel to the next, a slightly shifted sample gives a different answer. Measured here as the
// disagreement of the 3x3 neighbourhood (level-l pixels) in edge-ness e = sin^2(sem) (0 line, 1 edge), i.e. in what the
// Line / Edge view shows: strength-weighted standard deviation, scaled to 0..1 (max std of a 0..1 quantity is 0.5)
//   u = 2 sqrt(sum w e^2 / sum w - (sum w e / sum w)^2)
// Only e, not the angle 2 sem: at the +-90 deg orientation wrap the sign of sem flips between neighbours (both
// polarities of an edge are edges), which an angle measure (|mean exp(2i sem)|) wrongly counted as disagreement.
// out = (w * u, w): unsureness, strength-weighted for the bicubic upsampling (display.wgsl modes 21, 22).
// Headless check (face, 1-px shifts over 2^(L+1) frames; flicker = std of the displayed edge-ness over the frames):
// correlation with the flicker 0.81 (L2) / 0.82 (L3); the angle measure 0.75, wider windows (5x5, or the 3x3 result
// blurred / dilated) 0.57 - 0.76, the strength alone 0.2, the distance from the line / edge boundary about 0.
@compute @workgroup_size(8, 8)
fn sem_sure(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    let n = vec2<i32>(textureDimensions(src));
    var se = 0.0;
    var se2 = 0.0;
    var wsum = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = p + vec2<i32>(dx, dy);
            if (q.x >= 0 && q.y >= 0 && q.x < n.x && q.y < n.y) {
                let v = textureLoad(src, q, 0);
                let e = sin(v.x) * sin(v.x);
                se += v.y * e;
                se2 += v.y * e * e;
                wsum += v.y;
            }
        }
    }
    let m = se / max(wsum, 1e-12);
    let u = 2.0 * sqrt(max(se2 / max(wsum, 1e-12) - m * m, 0.0));
    let w = textureLoad(src, p, 0).y;
    textureStore(dst, p, vec4<f32>(w * u, w, 0.0, 0.0));
}
