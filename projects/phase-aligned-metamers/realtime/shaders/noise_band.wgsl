// Noise band of level s, as NoiseGenerator.generate_noise_band (default 8 x 8 grid, all 64 locations).
//
// Python sums 64 images, one per grid location (Hs, Ws) in 0..7 x 0..7. In each, impulses sit at
// (Hs + 8m, Ws + 8n) (m, n >= 0, 8m < full height, 8n < full width) and carry the 7x7 core of the steerable filters
// (correlation, so the kernel appears flipped around the impulse). Every impulse is steered, phase-shifted and
// scaled with the statistics at its own position; Python gets them with torch.roll and nearest upsampling, whose
// index arithmetic is reproduced here exactly. Per pixel and location, at most one impulse is within reach (|d| <= 3).
//
// Effect of the parameters (NoiseGenerator.randomize_stats, keep_* < 1): each statistic is blended with a random field,
//   intensity = keep * intensity + (1 - keep) * u * std(intensity of level s)
//   direction, phase: blended on the unit circle (direction in the doubled-angle domain)
// The random fields come from a hash of (pixel of level s, s, statistic) instead of torch.rand, so they stay the
// same from frame to frame and while a slider moves; web/test_webgpu.py --keep uses the same hash as reference.

struct BandParams {
    full_h: f32,     // full-resolution image height (size of the impulse maps)
    full_w: f32,
    level: f32,      // s
    randomize: f32,  // 1: blend with the random fields (some keep < 1)
    keep: vec4<f32>, // keep_intensity, keep_dir, keep_phase, unused
}

@group(0) @binding(0) var<uniform> BP: BandParams;
@group(0) @binding(1) var<storage, read> filt: array<f32, 567>;          // 7 filters x 9x9 (constants.json)
@group(0) @binding(2) var rep: texture_2d<f32>;                          // intensity_rep, dir_rep, phase_rep of level s
@group(0) @binding(3) var dst: texture_storage_2d<r32float, write>;      // noise band at the size of level s
@group(0) @binding(4) var<storage, read> rep_std: array<vec2<f32>, 8>;   // (std of intensity_rep, 1) per level

// PCG hash (Jarzynski & Olano 2020)
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
// uniform random number in [0, 1) for pixel p of level s and statistic k (0 intensity, 1 direction, 2 phase)
fn rand(p: vec2<i32>, s: u32, k: u32) -> f32 {
    let h = pcg(pcg(pcg(u32(p.x)) + u32(p.y)) + s * 4u + k);
    return f32(h >> 8u) / 16777216.0;
}
// angle of keep * e^(i a) + (1 - keep) * e^(i b)
fn cyclic_blend(a: f32, b: f32, keep: f32) -> f32 {
    let v = keep * vec2<f32>(cos(a), sin(a)) + (1.0 - keep) * vec2<f32>(cos(b), sin(b));
    return atan2(v.y, v.x);
}
// statistics of field pixel p (intensity, direction, phase), blended with the random fields if requested
fn stats_at(p: vec2<i32>) -> vec3<f32> {
    var f = textureLoad(rep, p, 0).xyz;
    if (BP.randomize > 0.5) {
        let s = u32(BP.level);
        let k = BP.keep;
        let pi = 3.14159265358979;
        f.x = k.x * f.x + (1.0 - k.x) * rand(p, s, 0u) * rep_std[s].x;
        f.y = cyclic_blend(2.0 * f.y, 2.0 * (rand(p, s, 1u) * pi - pi), k.y) / 2.0;
        f.z = cyclic_blend(f.z, rand(p, s, 2u) * 2.0 * pi - pi, k.z);
    }
    return f;
}

// offset of pixel coordinate v from the impulse of grid offset `shift` that reaches it, or 99 if none does
fn impulse_offset(v: i32, shift: i32, full: i32) -> i32 {
    let d = ((v - shift) % 8 + 8) % 8;
    if (d == 4) { return 99; }                         // between two impulses: out of reach of both (radius 3)
    let o = select(d - 8, d, d <= 3);                  // pixel - impulse position
    let m8 = v - o - shift;                            // 8m (or 8n)
    if (m8 < 0 || m8 >= full) { return 99; }           // no such impulse
    return o;
}

// field coordinate used for pixel coordinate v: torch.roll(shifts=-3 + shift) of the nearest-upsampled samples
// fields[shift::8]
fn field_coord(v: i32, shift: i32, size: i32) -> i32 {
    let n = (size - shift + 7) / 8;                    // number of samples in fields[shift::8]
    let idx = ((v + 3 - shift) % size + size) % size;
    let src = min(i32(floor(f32(idx) * (f32(n) / f32(size)))), n - 1);
    return shift + 8 * src;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let x = i32(id.x);
    let y = i32(id.y);
    let w = i32(dims.x);
    let h = i32(dims.y);

    var noise = 0.0;
    for (var loc = 0; loc < 64; loc++) {
        let ws = loc / 8;            // starting_x = [i for i in range(8) for _ in range(8)]
        let hs = loc % 8;            // starting_y = list(range(8)) * 8
        let oy = impulse_offset(y, hs, i32(BP.full_h));
        let ox = impulse_offset(x, ws, i32(BP.full_w));
        if (oy == 99 || ox == 99) { continue; }

        let f = stats_at(vec2<i32>(field_coord(x, ws, w), field_coord(y, hs, h)));
        let k = (4 - oy) * 9 + (4 - ox);             // 7x7 core [1:-1, 1:-1] of the 9x9 filters, flipped
        let c = cos(f.y);
        let s = sin(f.y);
        let G = c * c * filt[k] + (-2.0 * c * s) * filt[81 + k] + s * s * filt[162 + k];
        let H = c * c * c * filt[243 + k] + (-3.0 * c * c * s) * filt[324 + k]
              + (3.0 * c * s * s) * filt[405 + k] + (-s * s * s) * filt[486 + k];
        noise += (cos(f.z) * G - sin(f.z) * H) * f.x;
    }
    textureStore(dst, vec2<i32>(id.xy), vec4<f32>(noise, 0.0, 0.0, 0.0));
}
