// Shared by the repair shaders. Requires sampling.wgsl.

struct RepairParams {
    scale: f32,          // level s that is computed
    _p0: f32,
    align: f32,          // 1: align the orientation convention of complex responses before combining them (see align_sign)
    orient: f32,         // 1: orientation stability fix (web only): orientation upsampled weighted by its confidence
}

@group(0) @binding(0) var<uniform> RP: RepairParams;
@group(0) @binding(15) var<storage, read> FIT: array<f32, 12>;   // intensity modulators per level (fit.wgsl)

fn modulator() -> f32 { return FIT[u32(RP.scale)]; }

const PI: f32 = 3.14159265358979;

fn polar(r: f32, a: f32) -> vec2<f32> { return vec2<f32>(r * cos(a), r * sin(a)); }
fn carg(z: vec2<f32>) -> f32 { return atan2(z.y, z.x); }
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn conj(z: vec2<f32>) -> vec2<f32> { return vec2<f32>(z.x, -z.y); }

// Orientation is defined modulo pi, but the steered H filter is odd: H(theta + pi) = -H(theta). So (dir, phase) and
// (dir + pi, -phase) describe the same local signal, and where dir wraps from -pi/2 to +pi/2 the phase is conjugated.
// Before complex responses with different orientations are combined (interpolated, multiplied, blended), each is
// brought into the convention of a reference orientation: conjugated where its orientation points the other way.
// Returns the sign to apply to the phase (or to the imaginary part). RP.align = 0 keeps the old (unaligned) behaviour.
fn align_sign(theta: f32, theta_ref: f32) -> f32 {
    if (RP.align < 0.5) { return 1.0; }
    return select(1.0, -1.0, cos(theta - theta_ref) < 0.0);
}

// source position of target pixel p when upsampling by 2 from a texture of size src (isu.upsample)
fn pos2(p: vec2<i32>, src: vec2<u32>) -> vec2<f32> {
    return vec2<f32>(upsample_pos(f32(p.x), 2.0, f32(src.x)), upsample_pos(f32(p.y), 2.0, f32(src.y)));
}

// Complex bicubic interpolation of polar(mag_tex.b, ang_tex.z) at pos, i.e. isu.upsample applied to the real and
// imaginary part of the complex field |G + iH| * exp(i * phase_rep). Each tap is aligned to orientation theta_ref
// (its own orientation: dir_rep = ang_tex.y).
fn bicubic_mag_phase(mag_tex: texture_2d<f32>, ang_tex: texture_2d<f32>, pos: vec2<f32>, theta_ref: f32) -> vec2<f32> {
    let dims = vec2<i32>(textureDimensions(mag_tex));
    let base = floor(pos);
    let t = pos - base;
    let wx = cubic_weights(t.x);
    let wy = cubic_weights(t.y);
    var acc = vec2<f32>(0.0);
    for (var j = 0; j < 4; j++) {
        let y = clamp(i32(base.y) - 1 + j, 0, dims.y - 1);
        var row = vec2<f32>(0.0);
        for (var i = 0; i < 4; i++) {
            let q = vec2<i32>(clamp(i32(base.x) - 1 + i, 0, dims.x - 1), y);
            let r = textureLoad(ang_tex, q, 0);
            row += wx[i] * polar(textureLoad(mag_tex, q, 0).b, align_sign(r.y, theta_ref) * r.z);
        }
        acc += wy[j] * row;
    }
    return acc;
}

// Fix 4, dense coarse evaluation (repair_phi.wgsl, repair_level.wgsl; web only): RP._p0 = 0 off, else cutoff + 1.
// Where a coarser level k is measured, its complex response is taken from its filters evaluated at every pixel of the
// finer grid (analysis_filters.wgsl main_dilated) instead of interpolated from its own pixels. Weight of the measured
// part of level k at pixel p of a grid of size grid_size: 1 at or above the cutoff (measured everywhere), else its mask.
fn dense_on() -> bool { return RP._p0 > 0.5; }
fn measured_weight(maps: texture_2d<f32>, p: vec2<i32>, grid_size: vec2<u32>, k: f32) -> f32 {
    if (k >= RP._p0 - 1.0) { return 1.0; }
    let full = textureDimensions(maps);
    let scale = vec2<f32>(f32(full.x) / f32(grid_size.x), f32(full.y) / f32(grid_size.y));
    let q = min(vec2<i32>(floor(vec2<f32>(p) * scale)), vec2<i32>(full) - 1);
    return 1.0 - clamp(textureLoad(maps, q, 0).y - k, 0.0, 1.0);
}

// Mask of level s at level pixel p (Worm.compute_small_masks): the full-resolution mask
// 1 - clamp(level - s, 0, 1), sampled with nearest neighbour (F.interpolate(mode='nearest')).
fn level_mask(maps: texture_2d<f32>, p: vec2<i32>, level_size: vec2<u32>) -> f32 {
    let full = textureDimensions(maps);
    let scale = vec2<f32>(f32(full.x) / f32(level_size.x), f32(full.y) / f32(level_size.y));
    let q = min(vec2<i32>(floor(vec2<f32>(p) * scale)), vec2<i32>(full) - 1);
    let L = textureLoad(maps, q, 0).y;
    return 1.0 - clamp(L - RP.scale, 0.0, 1.0);
}
