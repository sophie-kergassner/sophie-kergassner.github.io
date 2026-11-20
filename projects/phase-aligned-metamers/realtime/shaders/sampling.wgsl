// Shared sampling helpers. Prepended by the host where needed.

// PyTorch's F.pad(mode='reflect') index rule (the edge pixel is not repeated: ... c b | a b c ...).
fn reflect_torch(i: i32, n: i32) -> i32 {
    var j = i;
    if (j < 0) { j = -j; }
    if (j >= n) { j = 2 * (n - 1) - j; }
    return j;
}

// Source position (in source pixel units) of target pixel x when upsampling by `factor`, as isu.upsample computes it:
// grid = 2 * (x / factor) / (src_size - 1) - 1, then grid_sample(align_corners=True) maps it back with
// ((grid + 1) / 2) * (src_size - 1). Doing the same float operations keeps the result identical.
fn upsample_pos(x: f32, factor: f32, src_size: f32) -> f32 {
    let denom = max(src_size - 1.0, 1.0);
    let g = 2.0 * (x / factor) / denom - 1.0;
    return ((g + 1.0) / 2.0) * (src_size - 1.0);
}

// Bicubic interpolation exactly as F.grid_sample(mode='bicubic', padding_mode='border'):
// cubic convolution with A = -0.75; out-of-range taps are clamped to the border.
const CUBIC_A: f32 = -0.75;
fn cubic1(x: f32) -> f32 { return ((CUBIC_A + 2.0) * x - (CUBIC_A + 3.0)) * x * x + 1.0; }
fn cubic2(x: f32) -> f32 { return ((CUBIC_A * x - 5.0 * CUBIC_A) * x + 8.0 * CUBIC_A) * x - 4.0 * CUBIC_A; }
fn cubic_weights(t: f32) -> vec4<f32> {
    return vec4<f32>(cubic2(t + 1.0), cubic1(t), cubic1(1.0 - t), cubic2(2.0 - t));
}

// pos = (x, y) in source pixel units (pixel centres at integers)
fn bicubic(tex: texture_2d<f32>, pos: vec2<f32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(tex));
    let base = floor(pos);
    let t = pos - base;
    let wx = cubic_weights(t.x);
    let wy = cubic_weights(t.y);
    var acc = vec4<f32>(0.0);
    for (var j = 0; j < 4; j++) {
        let y = clamp(i32(base.y) - 1 + j, 0, dims.y - 1);
        var row = vec4<f32>(0.0);
        for (var i = 0; i < 4; i++) {
            let x = clamp(i32(base.x) - 1 + i, 0, dims.x - 1);
            row += wx[i] * textureLoad(tex, vec2<i32>(x, y), 0);
        }
        acc += wy[j] * row;
    }
    return acc;
}

// isu.upsample: value of the upsampled level at target pixel p (x = col, y = row), level `factor` times smaller.
fn upsample_at(tex: texture_2d<f32>, p: vec2<i32>, factor: f32) -> vec4<f32> {
    if (factor <= 1.0) { return textureLoad(tex, p, 0); }
    let src = vec2<f32>(textureDimensions(tex));
    let pos = vec2<f32>(upsample_pos(f32(p.x), factor, src.x), upsample_pos(f32(p.y), factor, src.y));
    return bicubic(tex, pos);
}
