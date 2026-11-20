// Final image (NoiseGenerator.generate_noise / get_noise_enhancement, isu.compose_rgb):
//   noise = sum over s < cutoff of  upsample(band_s) / std_s * modulator_s * clamp(level - s, 0, 1)
//   Y     = Y of the foveated base image + noise          (Cb, Cr unchanged)
// Requires common.wgsl and sampling.wgsl.

struct ComposeParams {
    cutoff: f32,
    base: f32,           // 0: per-pixel blur (img.fov), 1: pyramid foveation
    gain: f32,           // noise gain of the "Final image" view (1 = as in the pipeline, 0 = no noise)
    _p0: f32,
}

@group(0) @binding(1) var<uniform> CP: ComposeParams;
@group(0) @binding(2) var maps: texture_2d<f32>;
@group(0) @binding(3) var fov: texture_2d<f32>;
@group(0) @binding(4) var fov_pyr: texture_2d<f32>;
@group(0) @binding(5) var band0: texture_2d<f32>;
@group(0) @binding(6) var band1: texture_2d<f32>;
@group(0) @binding(7) var band2: texture_2d<f32>;
@group(0) @binding(8) var band3: texture_2d<f32>;
@group(0) @binding(9) var<storage, read> stds: array<f32, 8>;
@group(0) @binding(10) var dst: texture_storage_2d<rgba32float, write>;  // (Y + noise, Cb, Cr, noise)
@group(0) @binding(11) var<storage, read> FIT: array<f32, 12>;             // intensity modulators per level (fit.wgsl)

fn band_up(s: i32, p: vec2<i32>) -> f32 {
    switch s {
        case 0: { return textureLoad(band0, p, 0).r; }
        case 1: { return upsample_at(band1, p, 2.0).r; }
        case 2: { return upsample_at(band2, p, 4.0).r; }
        default: { return upsample_at(band3, p, 8.0).r; }
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    let L = textureLoad(maps, p, 0).y;
    var noise = 0.0;
    for (var s = 0; s < min(i32(CP.cutoff), 4); s++) {
        let w = clamp(L - f32(s), 0.0, 1.0);
        if (w > 0.0) {
            noise += band_up(s, p) / stds[s] * FIT[s] * w;
        }
    }
    noise *= CP.gain;
    var base = textureLoad(fov, p, 0);
    if (CP.base > 0.5) { base = textureLoad(fov_pyr, p, 0); }
    textureStore(dst, p, vec4<f32>(base.r + noise, base.g, base.b, noise));
}
