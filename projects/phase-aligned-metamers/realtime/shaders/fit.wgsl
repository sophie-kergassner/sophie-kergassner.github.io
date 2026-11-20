// Intensity fit (Worm.compute_intensity_modulators) on the GPU, so that it can follow per-frame analyses:
// weighted least-squares line std = m * level + b through the Laplacian band stds (lap_finalize.wgsl),
// intensity modulator of level i = max(m * i + b, 0).
//   weighted fit (default): all bands with weight > 0, each measured where its level is fully kept (mask == 1),
//                           weighted by that fraction of the image
//   cutoff fit (paper):     bands cutoff .. last, unweighted

struct FitParams {
    cutoff: f32,
    n_bands: f32,        // number of Laplacian bands
    weighted: f32,       // 1: weighted fit, 0: cutoff fit
    _p0: f32,
}

@group(0) @binding(0) var<uniform> FP: FitParams;
@group(0) @binding(1) var<storage, read> stats: array<vec2<f32>, 8>;   // (std, weight) per band
@group(0) @binding(2) var<storage, read_write> fit: array<f32, 12>;    // [0..7] modulators, [8] m, [9] b

fn weight(i: i32) -> f32 {
    if (FP.weighted > 0.5) { return stats[i].y; }
    return select(0.0, 1.0, i >= i32(FP.cutoff));
}

@compute @workgroup_size(1)
fn main() {
    let n = i32(FP.n_bands);
    var sw = 0.0;
    var mx = 0.0;
    var my = 0.0;
    for (var i = 0; i < n; i++) { let w = weight(i); sw += w; mx += w * f32(i); my += w * stats[i].x; }
    mx /= sw;
    my /= sw;
    var sxy = 0.0;
    var sxx = 0.0;
    for (var i = 0; i < n; i++) {
        let w = weight(i);
        sxy += w * (f32(i) - mx) * (stats[i].x - my);
        sxx += w * (f32(i) - mx) * (f32(i) - mx);
    }
    let m = select(0.0, sxy / sxx, sxx > 0.0);
    let b = my - m * mx;
    for (var i = 0; i < 8; i++) { fit[i] = max(m * f32(i) + b, 0.0); }
    fit[8] = m;
    fit[9] = b;
}
