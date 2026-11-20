// Shared by the repair shaders. Requires sampling.wgsl.

struct RepairParams {
    scale: f32,          // level s that is computed
    _p0: f32,
    _p1: f32,
    _p2: f32,
}

@group(0) @binding(0) var<uniform> RP: RepairParams;
@group(0) @binding(15) var<storage, read> FIT: array<f32, 12>;   // intensity modulators per level (fit.wgsl)

fn modulator() -> f32 { return FIT[u32(RP.scale)]; }

const PI: f32 = 3.14159265358979;

fn polar(r: f32, a: f32) -> vec2<f32> { return vec2<f32>(r * cos(a), r * sin(a)); }
fn carg(z: vec2<f32>) -> f32 { return atan2(z.y, z.x); }
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn conj(z: vec2<f32>) -> vec2<f32> { return vec2<f32>(z.x, -z.y); }

// source position of target pixel p when upsampling by 2 from a texture of size src (isu.upsample)
fn pos2(p: vec2<i32>, src: vec2<u32>) -> vec2<f32> {
    return vec2<f32>(upsample_pos(f32(p.x), 2.0, f32(src.x)), upsample_pos(f32(p.y), 2.0, f32(src.y)));
}

// Complex bicubic interpolation of polar(mag_tex.b, ang_tex.z) at pos, i.e. isu.upsample applied to the real and
// imaginary part of the complex field |G + iH| * exp(i * phase_rep).
fn bicubic_mag_phase(mag_tex: texture_2d<f32>, ang_tex: texture_2d<f32>, pos: vec2<f32>) -> vec2<f32> {
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
            row += wx[i] * polar(textureLoad(mag_tex, q, 0).b, textureLoad(ang_tex, q, 0).z);
        }
        acc += wy[j] * row;
    }
    return acc;
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
