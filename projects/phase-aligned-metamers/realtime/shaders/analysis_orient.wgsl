// Analysis, step 2 (per pyramid level): dominant orientation, steered responses, phase and magnitude.
//   C2, C3 blurred with a 5x5 box filter (zero padding)          phu.get_theta_d
//   dir   = atan2(C3, C2) / 2,  S = sqrt(C2^2 + C3^2)
//   G     = cos^2 Ga - 2 cos sin Gb + sin^2 Gc                   phu.get_Gk
//   H     = cos^3 Ha - 3 cos^2 sin Hb + 3 cos sin^2 Hc - sin^3 Hd  phu.get_Hk
//   phase = angle(G + iH),  magnitude = |G + iH|                 Worm.compute_steered_G_H / compute_phase

@group(0) @binding(0) var ra: texture_2d<f32>;                           // Ga, Gb, Gc, Ha
@group(0) @binding(1) var rb: texture_2d<f32>;                           // Hb, Hc, Hd, 0
@group(0) @binding(2) var cc: texture_2d<f32>;                           // C2, C3
@group(0) @binding(3) var dst: texture_storage_2d<rgba32float, write>;   // dir, phase, |G + iH|, S

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let n = vec2<i32>(dims);
    let p = vec2<i32>(id.xy);

    var C = vec2<f32>(0.0);
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let q = p + vec2<i32>(dx, dy);
            if (q.x >= 0 && q.y >= 0 && q.x < n.x && q.y < n.y) {   // zero padding
                C += textureLoad(cc, q, 0).rg;
            }
        }
    }
    C = C / 25.0;

    let theta = atan2(C.y, C.x) / 2.0;
    let S = sqrt(C.x * C.x + C.y * C.y);
    let c = cos(theta);
    let s = sin(theta);
    let a = textureLoad(ra, p, 0);
    let b = textureLoad(rb, p, 0);
    let G = c * c * a.x + (-2.0 * c * s) * a.y + s * s * a.z;
    let H = c * c * c * a.w + (-3.0 * c * c * s) * b.x + (3.0 * c * s * s) * b.y + (-s * s * s) * b.z;

    textureStore(dst, p, vec4<f32>(theta, atan2(H, G), sqrt(G * G + H * H), S));
}
