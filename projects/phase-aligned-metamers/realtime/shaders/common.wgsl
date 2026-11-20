// Shared parameters (one uniform buffer, written by the host every frame).
// Prepended to foveate.wgsl and display.wgsl.

struct Params {
    size: vec4<f32>,      // image width [px], image height [px], pixel pitch [px/cm], sigma0 of the pyramid kernel
    gaze: vec4<f32>,      // gaze row [px], gaze col [px], uniform level (< 0: off), highest stack layer
    viewer: vec4<f32>,    // viewer vertical [cm], horizontal [cm], distance [cm], view mode
    ratio_lo: vec4<f32>,  // sigma calibration ratio (measured / formula) at pyramid levels 0..3
    ratio_hi: vec4<f32>,  // ... at levels 4..7
    disp: vec4<f32>,      // canvas width [px], canvas height [px], show gaze marker (0/1), unused
    aux: vec4<f32>,       // display: pyramid level of the shown map, display scale, unused, cutoff
}

@group(0) @binding(0) var<uniform> P: Params;

// Pixel (row, col) -> position on the screen in cm relative to the image centre (pixel centres), as ViewConfig.px_to_cm.
fn px_to_cm(rc: vec2<f32>) -> vec2<f32> {
    return (rc - (vec2<f32>(P.size.y, P.size.x) - 1.0) * 0.5) / P.size.z;
}

// Eccentricity [deg] of pixel (row, col): angle at the eye between gaze direction and pixel direction (ViewConfig).
fn eccentricity(rc: vec2<f32>) -> f32 {
    let viewer = vec3<f32>(P.viewer.x, P.viewer.y, -P.viewer.z);
    let to_pixel = vec3<f32>(px_to_cm(rc), 0.0) - viewer;
    let to_gaze = vec3<f32>(px_to_cm(P.gaze.xy), 0.0) - viewer;
    let c = dot(to_pixel, to_gaze) / (length(to_pixel) * length(to_gaze) + 1e-8);
    return degrees(acos(clamp(c, -1.0, 1.0)));
}

// ViewConfig.map_eccentricity_to_pyr_level: piecewise linear, 0 / 2 / 3 / 4 at 4 / 8 / 16 / 24 deg.
fn ecc_to_level(e: f32) -> f32 {
    if (e < 4.0)  { return 0.0; }
    if (e < 8.0)  { return 2.0 * (e - 4.0) / 4.0; }
    if (e < 16.0) { return 2.0 + (e - 8.0) / 8.0; }
    if (e < 24.0) { return 3.0 + (e - 16.0) / 8.0; }
    return 4.0;
}

fn ratio_at(i: i32) -> f32 {
    let k = clamp(i, 0, 7);
    if (k < 4) { return P.ratio_lo[k]; }
    return P.ratio_hi[k - 4];
}

// isu.pyramid_level_to_sigma: formula sigma0 * sqrt((4^L - 1) / 3), scaled by the measured/formula ratio,
// interpolated linearly between integer levels.
fn level_to_sigma(level: f32) -> f32 {
    let L = clamp(level, 0.0, 7.0);
    let formula = P.size.w * sqrt((exp2(2.0 * L) - 1.0) / 3.0);   // 4^L; exp2 is more precise than pow on GPUs
    let i = i32(floor(L));
    let f = L - f32(i);
    let ratio = ratio_at(i) * (1.0 - f) + ratio_at(i + 1) * f;
    return formula * ratio;
}
