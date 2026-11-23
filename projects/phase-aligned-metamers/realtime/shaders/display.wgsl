// Draws the selected view to the canvas.
// View modes (P.viewer.w); "aux" is a map of pyramid level l = P.aux.x:
//    0 original, 1 foveated (per-pixel blur), 2 foveated (pyramid), 3 difference per-pixel blur - pyramid (Y),
//    4 eccentricity, 5 pyramid level, 6 blur sigma,
//    7 Laplacian band, 8 orientation, 9 phase, 10 magnitude |G + iH|       (measured)
//   11 pyramid image (RGB),
//   12 intensity_rep, 13 dir_rep, 14 phase_rep                             (after the repair)
//   19, 20 phase semantics as structure (line / edge / none), measured / repaired (aux: semantics.wgsl of level l;
//          in the form of sem_smooth.wgsl, upsampled bicubically)
//   21, 22 sureness of the Line / Edge judgement, measured / repaired (aux: sem_smooth.wgsl sem_sure of level l,
//          upsampled bicubically): sure (pale) -> unsure (dark), white where the band has little energy
//   15 final image, 16 noise enhancement,
//   18 contribution of level l to the noise enhancement (aux = noise band l):
//      upsample(band_l) / std_l * modulator_l * clamp(level - l, 0, 1), as one term of compose.wgsl
// Requires common.wgsl and sampling.wgsl (prepended by the host).

@group(0) @binding(1) var ycc: texture_2d<f32>;
@group(0) @binding(2) var fov: texture_2d<f32>;
@group(0) @binding(3) var maps: texture_2d<f32>;
@group(0) @binding(4) var fov_pyr: texture_2d<f32>;
@group(0) @binding(5) var aux: texture_2d<f32>;
@group(0) @binding(6) var final_img: texture_2d<f32>;              // (Y + noise, Cb, Cr, noise)
@group(0) @binding(7) var<storage, read> stds: array<f32, 8>;      // std of each upsampled noise band
@group(0) @binding(8) var<storage, read> FIT: array<f32, 12>;      // intensity modulators per level (fit.wgsl)

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VSOut {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));   // (0,0), (2,0), (0,2): one triangle covering the screen
    var out: VSOut;
    out.pos = vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(xy.x, 1.0 - xy.y);
    return out;
}

// isu.ycrcb_2_rgb
fn ycc_to_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + 1.403 * (c.y - 0.5),
                     c.x - 0.714 * (c.y - 0.5) - 0.344 * (c.z - 0.5),
                     c.x + 1.773 * (c.z - 0.5));
}

// polynomial approximation of matplotlib's 'inferno' colormap
fn inferno(t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    let c0 = vec3<f32>(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
    let c1 = vec3<f32>(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
    let c2 = vec3<f32>(11.60249308247187, -3.972853965665698, -15.9423941062914);
    let c3 = vec3<f32>(-41.70399613139459, 17.43639888205313, 44.35414519872813);
    let c4 = vec3<f32>(77.162935699427, -33.40235894210092, -81.80730925738993);
    let c5 = vec3<f32>(-71.31942824499214, 32.62606426397723, 73.20951985803202);
    let c6 = vec3<f32>(25.13112622477341, -12.24266895238567, -23.07032500287172);
    return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), vec3<f32>(0.0), vec3<f32>(1.0));
}

// sequential colour map from the project colours: near-black #1d1216 -> dark crimson #6e072c -> crimson #bf0c4c -> salmon #dd8a78 -> beige #f1efe3, brightness increasing throughout (linear in sRGB); same as warm() in main.js (legend)
fn warm(t_in: f32) -> vec3<f32> {
    var stops = array<vec3<f32>, 5>(
        vec3<f32>(0.114, 0.071, 0.086),    // #1d1216
        vec3<f32>(0.431, 0.027, 0.173),    // #6e072c
        vec3<f32>(0.749, 0.047, 0.298),    // #bf0c4c
        vec3<f32>(0.867, 0.541, 0.471),    // #dd8a78
        vec3<f32>(0.945, 0.937, 0.890));   // #f1efe3
    let x = clamp(t_in, 0.0, 1.0) * 4.0;
    let i = min(i32(floor(x)), 3);
    return mix(stops[i], stops[i + 1], x - f32(i));
}

// diverging colour map from the project colours: slate blue #5b7088 (-1) - beige #f1efe3 (0) - crimson #bf0c4c (+1); as diverging() in main.js
fn diverging(v_in: f32) -> vec3<f32> {
    let v = clamp(v_in, -1.0, 1.0);
    let mid = vec3<f32>(0.945, 0.937, 0.890);
    if (v < 0.0) { return mix(mid, vec3<f32>(0.357, 0.439, 0.533), -v); }
    return mix(mid, vec3<f32>(0.749, 0.047, 0.298), v);
}

// cyclic colormap for angles (t and t + 1 give the same colour), from the project colours, around the circle: beige #f1efe3 -> crimson #bf0c4c -> dark #2b1a22 -> slate blue #5b7088 -> beige (linear in sRGB);
// same as cyclic() in main.js (legend)
fn cyclic(t: f32) -> vec3<f32> {
    var stops = array<vec3<f32>, 4>(
        vec3<f32>(0.945, 0.937, 0.890),    // beige   #f1efe3
        vec3<f32>(0.749, 0.047, 0.298),    // crimson #bf0c4c
        vec3<f32>(0.169, 0.102, 0.133),    // dark    #2b1a22
        vec3<f32>(0.357, 0.439, 0.533));   // slate   #5b7088
    let x = fract(t) * 4.0;
    let i = min(i32(floor(x)), 3);
    return mix(stops[i], stops[(i + 1) % 4], x - f32(i));
}

// Which structure the phase (here: the phase semantics, semantics.wgsl) describes. phase = atan2(H, G): near 0 / pi the even filter G answers alone -> a line
// (bright / dark), near +-pi/2 the odd filter H -> an edge (rising / falling); the sign is ignored. Line: slate blue
// #5b7088, edge: crimson #bf0c4c, mixed by sin^2(phase). Where the band has little energy the phase is meaningless:
// faded to white (beige #f1efe3) by the strength t (1 = top of the intensity scale; full colour from t = 0.5 on).
// As structureColor() in main.js (legend).
fn structure(phi: f32, t: f32) -> vec3<f32> {
    let s = sin(phi);
    let hue = mix(vec3<f32>(0.357, 0.439, 0.533), vec3<f32>(0.749, 0.047, 0.298), s * s);
    return mix(vec3<f32>(0.945, 0.937, 0.890), hue, clamp(2.0 * t, 0.0, 1.0));
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
    let size = vec2<f32>(P.size.x, P.size.y);
    let p = vec2<i32>(clamp(floor(in.uv * size), vec2<f32>(0.0), size - 1.0));
    let mode = i32(P.viewer.w);
    let m = textureLoad(maps, p, 0);
    let l = i32(P.aux.x);
    // pixel of pyramid level l: level l keeps every 2^l-th pixel
    let lp = min(p >> vec2<u32>(u32(l)), vec2<i32>(textureDimensions(aux)) - 1);
    let a = textureLoad(aux, lp, 0);
    let fin = textureLoad(final_img, p, 0);

    var rgb: vec3<f32>;
    switch mode {
        case 0: { rgb = ycc_to_rgb(textureLoad(ycc, p, 0).rgb); }
        case 1: { rgb = ycc_to_rgb(textureLoad(fov, p, 0).rgb); }
        case 2: { rgb = ycc_to_rgb(textureLoad(fov_pyr, p, 0).rgb); }
        case 3: { rgb = diverging((textureLoad(fov, p, 0).r - textureLoad(fov_pyr, p, 0).r) * P.aux.y); }
        case 4: { rgb = inferno(m.x / 40.0); }
        case 5: { rgb = inferno(m.y / 4.0); }
        case 6: { rgb = inferno(m.z / 10.0); }
        case 7: { rgb = vec3<f32>(0.5 + a.r * P.aux.y); }
        case 8, 13: { rgb = cyclic(select(a.x, a.y, mode == 13) / 3.14159265 + 0.5); }   // orientation, period pi
        case 9, 14: { rgb = cyclic(select(a.y, a.z, mode == 14) / 6.2831853 + 0.5); }    // phase, period 2 pi
        case 10: { rgb = warm(a.z * P.aux.y); }
        case 11: { rgb = ycc_to_rgb(a.rgb); }
        case 12: { rgb = warm(a.x * P.aux.y); }
        // phase semantics (aux = semantics.wgsl output: (sem, strength)), 19 measured / 20 repaired; the strength is
        // scaled as in the intensity views (10 / 12)
        case 19, 20: {
            // sem_smooth.wgsl form, upsampled bicubically: (strength * cos 2 sem, strength)
            let z = upsample_at(aux, p, exp2(f32(l))).xy;
            let c2 = clamp(z.x / max(z.y, 1e-12), -1.0, 1.0);   // averaged cos(2 sem): sin^2 of 0.5 acos(c2) = (1 - c2) / 2
            rgb = structure(0.5 * acos(c2), max(z.y, 0.0) * P.aux.y);
        }
        case 21, 22: {   // (strength * unsureness, strength), upsampled bicubically
            let z = upsample_at(aux, p, exp2(f32(l))).xy;
            let u = clamp(z.x / max(z.y, 1e-12), 0.0, 1.0);
            let hue = mix(vec3<f32>(0.70, 0.80, 0.76), vec3<f32>(0.16, 0.10, 0.22), u);
            rgb = mix(vec3<f32>(0.945, 0.937, 0.890), hue, clamp(2.0 * max(z.y, 0.0) * P.aux.y, 0.0, 1.0));
        }
        case 15: { rgb = ycc_to_rgb(fin.rgb); }
        case 16: { rgb = vec3<f32>(0.5 + fin.a * P.aux.y); }
        case 18: {
            var c = 0.0;
            if (l < i32(P.aux.w)) {   // P.aux.w: cutoff; levels >= cutoff are not synthesized
                let up = upsample_at(aux, p, exp2(f32(l))).r;
                c = up / max(stds[l], 1e-12) * FIT[l] * clamp(m.y - f32(l), 0.0, 1.0) * P.disp.w;   // x noise amp.
            }
            rgb = vec3<f32>(0.5 + c * P.aux.y);
        }
        default: { rgb = vec3<f32>(0.0); }
    }

    // gaze marker: beige ring (radius 7, 2 canvas px wide, anti-aliased) with a soft drop shadow (offset 1 px down,
    // blur ~3 px, 35 % black), like the overlay buttons; measured in canvas px at the continuous position
    if (P.disp.z > 0.5) {
        let px_per_canvas = P.size.x / P.disp.x;
        let rc = vec2<f32>(in.uv.y * size.y, in.uv.x * size.x) - 0.5;              // (row, col), continuous
        let ring = abs(distance(rc, P.gaze.xy) / px_per_canvas - 7.0) - 1.0;        // signed distance to the ring
        let shadow_d = abs(distance(rc - vec2<f32>(px_per_canvas, 0.0), P.gaze.xy) / px_per_canvas - 7.0) - 1.0;
        let shadow = 0.35 * exp(-pow(max(shadow_d, 0.0) / 3.0, 2.0));
        rgb = mix(rgb, vec3<f32>(0.0), shadow);
        rgb = mix(rgb, vec3<f32>(0.945, 0.937, 0.890), clamp(0.5 - ring, 0.0, 1.0));   // beige #f1efe3
    }
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
