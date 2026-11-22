// Phase-aligned metamers: WebGPU version of the full pipeline.
//   per image: YCbCr, blur stack, Gaussian/Laplacian pyramid and analysis of the original image
//   per frame: eccentricity -> level -> sigma, foveation, [pyramid + analysis of the foveated image],
//              Laplacian std + intensity fit, repair, noise synthesis, composition
// The host logic mirrors web/test_webgpu.py, which checks every shader against the Python package.

// Physical size of the image on the screen: the viewer measures the image's width with a ruler and enters it. That
// gives the size of one device pixel (cm per CSS px x devicePixelRatio), which stays the same when the window is
// resized, the page or the OS is zoomed, or the image goes full screen, so the width follows all of these.
// Until then: a guess for a 60 cm wide monitor (about 27") at the detected resolution.
const DEFAULT_CM_PER_DEVICE_PX = 60 / (screen.width * (window.devicePixelRatio || 1));
const MAX_PYR_DEPTH = 5;       // as in example.ipynb: cutoff <= 4
const ANALYSIS_LEVELS = MAX_PYR_DEPTH + 1;   // levels 0..5 (cutoff + 1 extra level for the phase repair)
const NOISE_LEVELS = MAX_PYR_DEPTH - 1;      // noise bands for levels 0..3 (s < cutoff <= 4)
const MAG_MAX = 0.8;                          // |G + iH| shown at the top of the colour scale (intensity views)
const MIN_LEVEL_SIZE = 8;      // Worm.MIN_LEVEL_SIZE

// Pipeline stages (flowchart). Each view: display mode (display.wgsl), level = which levels the slider offers,
// stat = statistic with the measured / repaired switch, legend = colour scale.
const FLOW = [
    { title: 'View. Geometry', views: [
        { name: 'Eccentricity', mode: 4, legend: { map: 'inferno', min: 0, max: 40, unit: 'deg' } },
        { name: 'Pyr. Level', mode: 5, legend: { map: 'inferno', min: 0, max: 4, unit: 'Pyramid Level' } },
        { name: 'Blur σ', mode: 6, legend: { map: 'inferno', min: 0, max: 10, unit: 'px' } },
    ] },
    { title: 'Foveation', views: [
        { name: 'Original', mode: 0, noGaze: true },
        { name: 'Foveated', fov: { blur: 1, pyramid: 2, diff: 3 } },   // per-pixel blur / from the pyramid / difference (toggle on the image)
    ] },
    { title: 'Image Pyramids', toFoveated: true, views: [
        { name: 'Image', mode: 11, level: 'pyramid' },
        { name: 'Laplacian Band', mode: 7, level: 'laplace', note: () => state.analyseFoveated
            ? 'Note how the fine levels (L0, L1, …) only have content around the gaze point: in the periphery they are plain grey, '
              + 'without detail. This lost detail is what our enhancement recovers.'
            : 'These are the bands of the full-res original, so they have detail everywhere. Switch "Measure Statistics On" to '
              + 'Foveated to see how foveation removes the fine levels in the periphery, the detail our enhancement recovers.',
          legend: { map: 'gray', scale: true, unit: 'Y' } },
    ] },
    { title: 'Statistics', toFoveated: true, views: [
        { name: 'Intensity', stat: { measured: 10, repaired: 12 }, level: 'analysis', legend: { map: 'warm', scale: true, positive: true, unit: '|G + iH| · Modulator' } },
        { name: 'Orientation', stat: { measured: 8, repaired: 13 }, level: 'analysis', legend: { map: 'cyclic', min: -90, max: 90, unit: 'deg' } },
        { name: 'Phase', stat: { measured: 9, repaired: 14 }, level: 'analysis', legend: { map: 'cyclic', min: -180, max: 180, unit: 'deg' },
          // "Line / Edge" toggle (#phaseBox): which structure the phase describes (structure() in display.wgsl)
          structure: { measured: 19, repaired: 20 },
          structureLegend: { map: 'structure', labels: ['Line', 'Edge'], unit: '· white: no data' },
          note: () => 'Angle: frequency-adjusted per level. Line / Edge: the semantics, only spatially upscaled.' },
    ] },
    { title: 'Noise', views: [
        { name: 'Enhancement', mode: 16, levels: true, keep: true, amp: true, legend: { map: 'gray', scale: true, unit: 'Y' } },
    ] },
    { title: 'Result', views: [
        { name: 'Final Image', mode: 15, gain: true, keep: true, uniform: true },
    ] },
];

const $ = (id) => document.getElementById(id);
// The demo can be embedded in another page: the element with [data-pam-demo] may set data-base, the path
// (relative to the page) of the folder that contains shaders/ and constants.json.
const BASE = (document.querySelector('[data-pam-demo]') || { dataset: {} }).dataset.base || '';
let errorShown = false;
// The demo could not start (no WebGPU, no adapter, an error while setting up): the page's placeholder (#pamFallback)
// replaces the demo, with the reason in it. Errors later on (e.g. a lost device) only show in the status line.
let started = false;
function showFallback(msg) {
    const fallback = document.getElementById('pamFallback');
    if (!fallback) return false;
    const reason = fallback.querySelector('.pam-fallback-reason');
    if (reason) reason.textContent = msg;
    fallback.hidden = false;
    document.querySelector('[data-pam-demo]').hidden = true;
    return true;
}
const status = (msg, isError = false) => {
    if (errorShown && !isError) return;   // keep the first error visible
    errorShown = errorShown || isError;
    $('status').textContent = msg;
    $('status').classList.toggle('error', isError);
    const pop = document.getElementById('infoPop');   // the status lives in the (i) popover: show errors right away
    if (isError && pop) pop.hidden = false;
};

const state = {
    stage: 5, view: 0,      // selected flowchart node and view
    variant: 'repaired',    // statistics: measured / repaired
    phaseShow: 'angle',     // Phase view: the angle ('angle', cyclic map) or the structure it describes ('structure': line / edge / none)
    fovMethod: 'blur',      // foveated view: per-pixel blur / from the pyramid / their difference
    level: 0,               // pyramid level of the level views
    gaze: null,             // [row, col] in image pixels
    frozen: false,          // click on the image freezes the gaze
    pointerInside: false,
    autoGaze: true,         // slow automatic eye movement while the mouse is outside the image
    distance: 50,           // viewing distance [cm]
    uniformLevel: -1,       // Final Image: < 0 eccentricity-dependent foveation, else one level for the whole image
    showGaze: true,
    analyseFoveated: false, // statistics measured on the full-res original (as in the paper; default) or the foveated
                            // image; switches to foveated on entering the Image Pyramids or Statistics stage
    fitWeighted: true,      // intensity fit: weighted over all bands (web version) or from the cutoff on (paper)
    enhLevel: -1,           // Enhancement view: -1 = sum of all levels, s = contribution of level s only
    gain: 1,                // noise gain of the "Final image" view (1 = as computed, 0 = no noise)
    enhOn: true,            // Final Image: enhancement on (foveated + noise) / off (foveated image only)
    levelOn: [true, true, true, true],   // Enhancement / Final Image with "All": which levels' noise is included
    noiseFrom: 'rep',       // noise synthesized from the extrapolated (repaired, 'rep') or the measured ('meas') statistics
    keep: [1, 1, 1],        // effect of the parameters: share of the measured intensity, orientation, phase kept in the
                            // noise (1 = as measured, 0 = random; NoiseGenerator.generate_noise keep_*)
    gpuInfo: null,          // last read-back of the intensity fit and the Laplacian std (for info line and legends)
    cmPerDevicePx: DEFAULT_CM_PER_DEVICE_PX,   // physical size of one device pixel [cm], from the measured image width
    cssWidth: 0,            // width of the shown image on the page [CSS px]
};

async function loadText(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`could not load ${url} (${r.status})`);
    return r.text();
}

// ------------------------------------------------------------------ colour maps (as in display.wgsl), for legends
function inferno(t) {
    t = Math.min(Math.max(t, 0), 1);
    const c = [[0.0002189403691192265, 0.001651004631001012, -0.01948089843709184], [0.1065134194856116, 0.5639564367884091, 3.932712388889277],
        [11.60249308247187, -3.972853965665698, -15.9423941062914], [-41.70399613139459, 17.43639888205313, 44.35414519872813],
        [77.162935699427, -33.40235894210092, -81.80730925738993], [-71.31942824499214, 32.62606426397723, 73.20951985803202],
        [25.13112622477341, -12.24266895238567, -23.07032500287172]];
    return [0, 1, 2].map((k) => Math.min(Math.max(c[0][k] + t * (c[1][k] + t * (c[2][k] + t * (c[3][k] + t * (c[4][k] + t * (c[5][k] + t * c[6][k]))))), 0), 1));
}
// sequential colour map from the project colours: near-black #1d1216 -> dark crimson #6e072c -> crimson #bf0c4c -> salmon #dd8a78 -> beige #f1efe3, brightness increasing throughout (linear in sRGB); as warm() in display.wgsl
const WARM_STOPS = [[0.114, 0.071, 0.086], [0.431, 0.027, 0.173], [0.749, 0.047, 0.298], [0.867, 0.541, 0.471], [0.945, 0.937, 0.890]];
const warm = (t) => {
    const x = Math.min(Math.max(t, 0), 1) * 4, i = Math.min(Math.floor(x), 3), f = x - i;
    return WARM_STOPS[i].map((v, k) => v + (WARM_STOPS[i + 1][k] - v) * f);
};
// diverging colour map from the project colours: slate blue #5b7088 (-1) - beige #f1efe3 (0) - crimson #bf0c4c (+1); as diverging() in display.wgsl (t in [0, 1])
const diverging = (t) => {
    const v = Math.min(Math.max(2 * t - 1, -1), 1), mid = [0.945, 0.937, 0.890];
    const end = v < 0 ? [0.357, 0.439, 0.533] : [0.749, 0.047, 0.298];
    return mid.map((m, k) => m + (end[k] - m) * Math.abs(v));
};
// legend of the difference view (Foveated: Difference)
const DIFF_LEGEND = { map: 'diverging', scale: true, unit: 'Y (Blur − Pyramid)' };
// cyclic colour map for angles, from the project colours, around the circle: beige #f1efe3 -> crimson #bf0c4c -> dark #2b1a22 -> slate blue #5b7088 -> beige (linear in sRGB); as cyclic() in display.wgsl
const CYCLIC_STOPS = [[0.945, 0.937, 0.890], [0.749, 0.047, 0.298], [0.169, 0.102, 0.133], [0.357, 0.439, 0.533]];
const cyclic = (t) => {
    const x = (t - Math.floor(t)) * 4, i = Math.min(Math.floor(x), 3), f = x - i;
    const a = CYCLIC_STOPS[i], b = CYCLIC_STOPS[(i + 1) % 4];
    return a.map((v, k) => v + (b[k] - v) * f);
};
// phase as structure, at full strength: line (slate blue #5b7088, t = 0) -> edge (crimson #bf0c4c, t = 1), mixed by
// sin^2(phase); as structure() in display.wgsl (which also fades to white where the band has little energy)
const structureColor = (t) => [0.357, 0.439, 0.533].map((v, k) => v + ([0.749, 0.047, 0.298][k] - v) * t);
function gradient(map) {
    const stops = [];
    for (let i = 0; i <= 16; i++) {
        const t = i / 16;
        const rgb = map === 'inferno' ? inferno(t) : map === 'cyclic' ? cyclic(t) : map === 'warm' ? warm(t) : map === 'diverging' ? diverging(t)
            : map === 'structure' ? structureColor(t) : [t, t, t];
        stops.push(`rgb(${rgb.map((x) => Math.round(255 * x)).join(',')}) ${(100 * t).toFixed(1)}%`);
    }
    return `linear-gradient(to right, ${stops.join(', ')})`;
}
const fmt = (v) => (Math.abs(v) >= 100 || v === 0 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toPrecision(3) : v.toPrecision(2));

async function main() {
    // ?demo=fallback in the URL: behave as without WebGPU (to check the placeholder)
    if (!navigator.gpu || new URLSearchParams(location.search).get('demo') === 'fallback') {
        if (!showFallback('This browser does not support WebGPU, which the real-time demo needs.')) {
            status('This browser does not support WebGPU. Use a current Chrome, Edge or Safari (or Firefox on Windows).', true);
        }
        return;
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
        const msg = 'No WebGPU adapter found (WebGPU may be switched off, or the graphics driver is not supported).';
        if (!showFallback(msg)) status(msg, true);
        return;
    }
    const device = await adapter.requestDevice();
    device.lost.then((info) => status(`GPU device lost: ${info.message}`, true));
    device.addEventListener('uncapturederror', (e) => status(`WebGPU error: ${e.error.message}`, true));

    const names = ['common', 'sampling', 'ycbcr', 'blur', 'foveate', 'foveate_pyramid', 'pyramid_down', 'laplace',
                   'analysis_filters', 'analysis_orient', 'lap_stats', 'lap_finalize', 'fit', 'repair_common', 'repair_init', 'repair_phi', 'repair_level', 'semantics',
                   'noise_band', 'noise_stats', 'noise_finalize', 'compose', 'display'];
    const src = Object.fromEntries(await Promise.all(names.map(async (n) => [n, await loadText(`${BASE}shaders/${n}.wgsl`)])));
    const consts = await (await fetch(`${BASE}constants.json`)).json();
    const code = (...parts) => parts.map((p) => src[p]).join('\n');

    // ---------------------------------------------------------------- pipelines (layouts as in test_webgpu.py)
    const C = GPUShaderStage.COMPUTE, F = GPUShaderStage.FRAGMENT;
    const tex = (binding, sampleType = 'unfilterable-float', viewDimension = '2d', visibility = C) =>
        ({ binding, visibility, texture: { sampleType, viewDimension } });
    const texs = (from, to, visibility = C) => Array.from({ length: to - from + 1 }, (_, i) => tex(from + i, 'unfilterable-float', '2d', visibility));
    const storage = (binding, format) =>
        ({ binding, visibility: C, storageTexture: { access: 'write-only', format, viewDimension: '2d' } });
    const uniform = (binding, visibility = C) => ({ binding, visibility, buffer: { type: 'uniform' } });
    const roStorage = (binding, visibility = C) => ({ binding, visibility, buffer: { type: 'read-only-storage' } });
    const rwStorage = (binding) => ({ binding, visibility: C, buffer: { type: 'storage' } });

    const computePipeline = (shaderCode, entries, entryPoint = 'main') => {
        const layout = device.createBindGroupLayout({ entries });
        const pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ code: shaderCode }), entryPoint },
        });
        return { pipeline, layout };
    };

    const P = {
        ycbcr: computePipeline(code('ycbcr'), [tex(0, 'float'), storage(1, 'rgba32float')]),
        blurV: computePipeline(code('blur'), [uniform(0), tex(1), storage(2, 'rgba32float')], 'blur_v'),
        blurH: computePipeline(code('blur'), [uniform(0), tex(1), storage(2, 'rgba32float'), storage(3, 'rgba16float')], 'blur_h'),
        down: computePipeline(code('sampling', 'pyramid_down'), [roStorage(0), tex(1), storage(2, 'rgba32float')]),
        laplace: computePipeline(code('sampling', 'laplace'), [tex(0), tex(1), storage(2, 'r32float')]),
        filters: computePipeline(code('sampling', 'analysis_filters'),
            [roStorage(0), tex(1), storage(2, 'rgba32float'), storage(3, 'rgba32float'), storage(4, 'rgba32float')]),
        orient: computePipeline(code('analysis_orient'), [...texs(0, 2), storage(3, 'rgba32float')]),
        foveate: computePipeline(code('common', 'foveate'),
            [uniform(0), tex(1), tex(2, 'unfilterable-float', '2d-array'), storage(3, 'rgba32float'), storage(4, 'rgba32float')]),
        foveatePyr: computePipeline(code('common', 'sampling', 'foveate_pyramid'), [uniform(0), ...texs(1, 6), storage(7, 'rgba32float')]),
        stats: computePipeline(code('sampling', 'noise_stats'), [uniform(0), tex(1), tex(2), rwStorage(3)]),
        finalize: computePipeline(code('noise_finalize'), [uniform(0), roStorage(1), rwStorage(2)]),
        lapStats: computePipeline(code('lap_stats'), [uniform(0), tex(1), tex(2), rwStorage(3)]),
        lapFinalize: computePipeline(code('lap_finalize'), [uniform(0), roStorage(1), rwStorage(2)]),
        fit: computePipeline(code('fit'), [uniform(0), roStorage(1), rwStorage(2)]),
        repInit: computePipeline(code('sampling', 'repair_common', 'repair_init'), [uniform(0), tex(1), storage(2, 'rgba32float'), roStorage(15)]),
        repPhi: computePipeline(code('sampling', 'repair_common', 'repair_phi'), [uniform(0), ...texs(1, 4), storage(5, 'rgba32float')]),
        repLevel: computePipeline(code('sampling', 'repair_common', 'repair_level'), [uniform(0), ...texs(1, 5), storage(6, 'rgba32float'), roStorage(15)]),
        sem: computePipeline(code('sampling', 'repair_common', 'semantics'), [uniform(0), ...texs(1, 4), storage(5, 'rgba32float')]),
        band: computePipeline(code('noise_band'), [uniform(0), roStorage(1), tex(2), storage(3, 'r32float'), roStorage(4)]),
        compose: computePipeline(code('common', 'sampling', 'compose'),
            [uniform(0), uniform(1), ...texs(2, 8), roStorage(9), storage(10, 'rgba32float'), roStorage(11)]),
    };

    const canvas = $('canvas');
    const context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: canvasFormat, alphaMode: 'opaque' });
    const displayLayout = device.createBindGroupLayout({ entries: [uniform(0, F), ...texs(1, 6, F), roStorage(7, F), roStorage(8, F)] });
    const displayModule = device.createShaderModule({ code: code('common', 'sampling', 'display') });
    const displayPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [displayLayout] }),
        vertex: { module: displayModule, entryPoint: 'vs' },
        fragment: { module: displayModule, entryPoint: 'fs', targets: [{ format: canvasFormat }] },
        primitive: { topology: 'triangle-list' },
    });

    // ---------------------------------------------------------------- buffers
    const uniformBuffer = (bytes) => device.createBuffer({ size: bytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const fixedBuffer = (values, usage) => {
        const b = device.createBuffer({ size: values.length * 4, usage, mappedAtCreation: true });
        new Float32Array(b.getMappedRange()).set(values);
        b.unmap();
        return b;
    };
    const fixedUniform = (values) => fixedBuffer(values, GPUBufferUsage.UNIFORM);
    const paramsBuffer = uniformBuffer(28 * 4);
    const composeBuffer = uniformBuffer(16);
    const fitParamsBuffer = uniformBuffer(16);
    const repairBuffers = Array.from({ length: ANALYSIS_LEVELS }, (_, s) => fixedUniform([s, 0, 0, 0]));
    // phase semantics (semantics.wgsl) of level l from the measured (1) or the repaired (0) phases
    const semBuffers = { measured: repairBuffers.map((_, l) => fixedUniform([l, 1, 0, 0])), repaired: repairBuffers };
    const kernelBuffer = fixedBuffer(consts.pyr_kernel, GPUBufferUsage.STORAGE);
    const filterBuffer = fixedBuffer(consts.filters, GPUBufferUsage.STORAGE);
    const bandStdsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE });
    const lapStatsBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });   // (std, weight) per band
    const fitBuffer = device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const repStdBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE });   // (std of intensity_rep, 1) per level
    const readback = device.createBuffer({ size: 112, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    let readbackBusy = false;

    // ---------------------------------------------------------------- helpers
    const bindGroup = (layout, resources) => device.createBindGroup({
        layout,
        entries: resources.map(([binding, r]) => ({ binding, resource: r instanceof GPUBuffer ? { buffer: r } : r.createView ? r.createView() : r })),
    });
    const bg = (p, list) => bindGroup(p.layout, list.map((r, i) => [i, r]));   // bindings 0..n-1
    const run = (pass, p, group, w, h) => {   // one dispatch, 8 x 8 workgroups
        pass.setPipeline(p.pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
    };
    const runGroups = (pass, p, group, gx, gy) => {
        pass.setPipeline(p.pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(gx, gy);
    };
    const U = GPUTextureUsage;
    const RW = U.TEXTURE_BINDING | U.STORAGE_BINDING | U.COPY_SRC;

    // same as common.wgsl / ViewConfig / isu.pyramid_level_to_sigma (used on the CPU for the cutoff)
    function levelToSigma(level) {
        const L = Math.min(Math.max(level, 0), 7);
        const i = Math.floor(L), f = L - i;
        const ratio = consts.ratios[i] * (1 - f) + consts.ratios[Math.min(i + 1, 7)] * f;
        return consts.sigma0 * Math.sqrt((Math.pow(4, L) - 1) / 3) * ratio;
    }
    // image pixels per cm on the monitor
    const shownDevicePx = () => state.cssWidth * (window.devicePixelRatio || 1);   // image width in device px
    const imageWidthCm = () => shownDevicePx() * state.cmPerDevicePx;
    const pxPerCm = () => img.w / Math.max(imageWidthCm(), 1e-3);
    function eccentricity(row, col) {
        const ppc = pxPerCm();
        const cm = (r, c) => [(r - (img.h - 1) / 2) / ppc, (c - (img.w - 1) / 2) / ppc];
        const [py, px] = cm(row, col), [gy, gx] = cm(state.gaze[0], state.gaze[1]);
        const a = [py, px, state.distance], b = [gy, gx, state.distance];   // vectors from the eye (centred viewer)
        const c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (Math.hypot(...a) * Math.hypot(...b) + 1e-8);
        return Math.acos(Math.min(1, Math.max(-1, c))) * 180 / Math.PI;
    }
    function eccToLevel(e) {
        if (e < 4) return 0;
        if (e < 8) return 2 * (e - 4) / 4;
        if (e < 16) return 2 + (e - 8) / 8;
        if (e < 24) return 3 + (e - 16) / 8;
        return 4;
    }
    // Worm.compute_cutoff_and_masks: the first level whose mask covers the whole image, i.e. ceil(max level).
    // The highest eccentricity lies on the image border, which is sampled here.
    function cutoff() {
        let maxLevel = 0;
        if (uniformLevel() >= 0) {
            maxLevel = uniformLevel();
        } else {
            const step = 4;
            for (let c = 0; c < img.w; c += step) {
                maxLevel = Math.max(maxLevel, eccToLevel(eccentricity(0, c)), eccToLevel(eccentricity(img.h - 1, c)));
            }
            for (let r = 0; r < img.h; r += step) {
                maxLevel = Math.max(maxLevel, eccToLevel(eccentricity(r, 0)), eccToLevel(eccentricity(r, img.w - 1)));
            }
            maxLevel = Math.max(maxLevel, eccToLevel(eccentricity(img.h - 1, img.w - 1)));
        }
        // the phase repair of level s needs levels s+1 and s+2
        return Math.min(Math.ceil(maxLevel - 1e-6), MAX_PYR_DEPTH - 1, img.nAnalysis - 2);
    }

    // ---------------------------------------------------------------- per image
    let img = null;

    async function loadImage(url) {
        status(`Loading ${url} ...`);
        const blob = await (await fetch(url)).blob();
        const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
        const w = bmp.width, h = bmp.height;
        const t0 = performance.now();

        if (img) { img.textures.forEach((t) => t.destroy()); img.buffers.forEach((b) => b.destroy()); }
        const textures = [], buffers = [];
        const make = (tw, th, format, usage = RW, layers = 1) => {
            const t = device.createTexture({ size: [tw, th, layers], format, usage });
            textures.push(t);
            return t;
        };
        const maxLayer = Math.floor(levelToSigma(4)) + 1;   // highest sigma (level 4) needs layers up to floor(sigma) + 1
        const nLevels = Math.floor(Math.log2(Math.min(w, h) / MIN_LEVEL_SIZE)) + 1;
        const nAnalysis = Math.min(ANALYSIS_LEVELS, nLevels);
        img = { w, h, maxLayer, nLevels, nAnalysis, textures, buffers };

        img.src = make(w, h, 'rgba8unorm', U.TEXTURE_BINDING | U.COPY_DST | U.RENDER_ATTACHMENT);
        device.queue.copyExternalImageToTexture({ source: bmp }, { texture: img.src }, [w, h]);
        img.ycc = make(w, h, 'rgba32float');
        const tmp = make(w, h, 'rgba32float'), ping = make(w, h, 'rgba32float'), pong = make(w, h, 'rgba32float');
        img.stack = make(w, h, 'rgba16float', U.TEXTURE_BINDING | U.STORAGE_BINDING, maxLayer + 1);
        for (const k of ['maps', 'fov', 'fovPyr', 'final']) img[k] = make(w, h, 'rgba32float');

        // level sizes
        img.sizes = [[w, h]];
        for (let i = 1; i < nLevels; i++) img.sizes.push(img.sizes[i - 1].map((v) => Math.ceil(v / 2)));

        // two sets of pyramid, Laplacian bands and analysis: of the original (once) and of the foveated image (per frame)
        const makeSet = (level0) => {
            const set = { levels: [level0], laplace: [], analysis: [], scratch: [] };
            for (let i = 1; i < nLevels; i++) set.levels.push(make(...img.sizes[i], 'rgba32float'));
            for (let i = 0; i < nLevels - 1; i++) set.laplace.push(make(...img.sizes[i], 'r32float'));
            for (let i = 0; i < nAnalysis; i++) {
                set.analysis.push(make(...img.sizes[i], 'rgba32float'));
                set.scratch.push([0, 1, 2].map(() => make(...img.sizes[i], 'rgba32float')));
            }
            return set;
        };
        const pyramidGroups = (set, level0) => ({   // bind groups to build the set from `level0`
            down: set.levels.map((l, i) => i === 0 ? null : bg(P.down, [kernelBuffer, i === 1 ? level0 : set.levels[i - 1], l])),
            laplace: set.laplace.map((lap, i) => bg(P.laplace, [i === 0 ? level0 : set.levels[i], set.levels[i + 1], lap])),
            filters: set.analysis.map((_, i) => bg(P.filters, [filterBuffer, i === 0 ? level0 : set.levels[i], ...set.scratch[i]])),
            orient: set.analysis.map((a, i) => bg(P.orient, [...set.scratch[i], a])),
        });
        img.orig = makeSet(img.ycc);
        img.fovSet = makeSet(img.fov);
        img.orig.build = pyramidGroups(img.orig, img.ycc);
        img.fovSet.build = pyramidGroups(img.fovSet, img.fov);   // base image: per-pixel blur, as the Python pipeline

        // ---- per-image computation: YCbCr, blur stack, pyramid and analysis of the original
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        run(pass, P.ycbcr, bg(P.ycbcr, [img.src, img.ycc]), w, h);
        let cur = img.ycc;
        for (let k = 1; k <= maxLayer; k++) {
            const sigma = Math.sqrt(k * k - (k - 1) * (k - 1));
            const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
            const ab = ub.getMappedRange();
            new Float32Array(ab, 0, 1)[0] = sigma;
            new Int32Array(ab, 4, 1)[0] = Math.trunc(2 * sigma + 0.5);
            ub.unmap();
            const next = cur === ping ? pong : ping;
            run(pass, P.blurV, bg(P.blurV, [ub, cur, tmp]), w, h);
            run(pass, P.blurH, bg(P.blurH, [ub, tmp, next, img.stack.createView({ dimension: '2d', baseArrayLayer: k, arrayLayerCount: 1 })]), w, h);
            cur = next;
        }
        buildPyramid(pass, img.orig, img.orig.build);
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        [tmp, ping, pong].forEach((t) => t.destroy());

        // ---- per-frame resources and bind groups
        img.rep = img.sizes.slice(0, nAnalysis).map((s) => make(...s, 'rgba32float'));
        img.phi = img.sizes.slice(0, nAnalysis).map((s) => make(...s, 'rgba32float'));
        img.sem = img.sizes.slice(0, nAnalysis - 1).map((s) => make(...s, 'rgba32float'));   // level l needs level l+1
        img.bands = img.sizes.slice(0, NOISE_LEVELS).map((s) => make(...s, 'r32float'));
        const partial = (gx, gy) => {
            const b = device.createBuffer({ size: gx * gy * 8, usage: GPUBufferUsage.STORAGE });
            buffers.push(b);
            return b;
        };
        const groups16 = ([sw, sh]) => [Math.ceil(sw / 16), Math.ceil(sh / 16)];

        img.bandU = img.bands.map(() => { const u = uniformBuffer(32); buffers.push(u); return u; });   // BandParams per level
        const lvViews = [0, 1, 2, 3, 4, 5].map((i) => img.orig.levels[Math.min(i, nLevels - 1)]);
        img.bg = {
            foveate: bg(P.foveate, [paramsBuffer, img.ycc, img.stack.createView({ dimension: '2d-array' }), img.maps, img.fov]),
            foveatePyr: bg(P.foveatePyr, [paramsBuffer, ...lvViews, img.fovPyr]),
            fit: bg(P.fit, [fitParamsBuffer, lapStatsBuffer, fitBuffer]),
            band: img.bands.map((b, s) => bg(P.band, [img.bandU[s], filterBuffer, img.rep[s], b, repStdBuffer])),
            bandStats: img.bands.map((b, s) => {
                const [gx, gy] = groups16([w, h]), part = partial(gx, gy);
                return { groups: [gx, gy],
                         stats: bg(P.stats, [fixedUniform([2 ** s, gx, 0, 0]), b, img.ycc, part]),
                         finalize: bg(P.finalize, [fixedUniform([gx * gy, w * h, s, 0]), part, bandStdsBuffer]) };
            }),
            compose: bg(P.compose, [paramsBuffer, composeBuffer, img.maps, img.fov, img.fovPyr,
                ...[0, 1, 2, 3].map((s) => img.bands[Math.min(s, img.bands.length - 1)]), bandStdsBuffer, img.final, fitBuffer]),
        };
        // bind groups that depend on the analysed set
        const lapParts = img.sizes.slice(0, nLevels - 1).map((s) => {
            const [gx, gy] = groups16(s);
            const b = device.createBuffer({ size: gx * gy * 16, usage: GPUBufferUsage.STORAGE });
            buffers.push(b);
            return { groups: [gx, gy], buf: b, statsU: uniformBuffer(16), finalizeU: uniformBuffer(16) };
        });
        img.lapParts = lapParts;
        // effect of the parameters: plain std of intensity_rep per noise level (lap_stats / lap_finalize, unweighted)
        img.bg.repStd = img.bands.map((_, s) => ({
            groups: lapParts[s].groups,
            stats: bg(P.lapStats, [fixedUniform([s, lapParts[s].groups[0], 0, 0]), img.rep[s], img.maps, lapParts[s].buf]),
            finalize: bg(P.lapFinalize, [fixedUniform([lapParts[s].groups[0] * lapParts[s].groups[1], s, 0, 0]), lapParts[s].buf, repStdBuffer]),
        }));
        for (const set of [img.orig, img.fovSet]) {
            set.lapStats = set.laplace.map((lap, i) => ({
                groups: lapParts[i].groups,
                stats: bg(P.lapStats, [lapParts[i].statsU, lap, img.maps, lapParts[i].buf]),
                finalize: bg(P.lapFinalize, [lapParts[i].finalizeU, lapParts[i].buf, lapStatsBuffer]),
            }));
            set.repInit = img.rep.map((r, s) => bindGroup(P.repInit.layout, [[0, repairBuffers[s]], [1, set.analysis[s]], [2, r], [15, fitBuffer]]));
            set.repPhi = img.rep.map((_, s) => s + 2 < nAnalysis ? bg(P.repPhi, [repairBuffers[s],
                set.analysis[s + 1], img.rep[s + 1], set.analysis[s + 2], img.rep[s + 2], img.phi[s + 1]]) : null);
            set.repLevel = img.rep.map((r, s) => s + 1 < nAnalysis ? bindGroup(P.repLevel.layout, [[0, repairBuffers[s]], [1, img.maps],
                [2, set.analysis[s]], [3, img.rep[s + 1]], [4, set.analysis[s + 1]], [5, img.phi[s + 1]], [6, r], [15, fitBuffer]]) : null);
            set.sem = Object.fromEntries(['measured', 'repaired'].map((variant) => [variant, img.sem.map((out, l) => bg(P.sem,
                [semBuffers[variant][l], set.analysis[l], img.rep[l], set.analysis[l + 1], img.rep[l + 1], out]))]));
        }

        state.gaze = [h / 2, w / 2];
        state.gpuInfo = null;
        updateControls();
        resizeCanvas();
        status(`${w} × ${h} px · per-image precomputation in ${(performance.now() - t0).toFixed(0)} ms`);
        requestRender();
    }

    function buildPyramid(pass, set, groups) {   // Gaussian pyramid, Laplacian bands and analysis of one set
        for (let i = 1; i < img.nLevels; i++) run(pass, P.down, groups.down[i], ...img.sizes[i]);
        for (let i = 0; i < img.nLevels - 1; i++) run(pass, P.laplace, groups.laplace[i], ...img.sizes[i]);
        for (let i = 0; i < img.nAnalysis; i++) {
            run(pass, P.filters, groups.filters[i], ...img.sizes[i]);
            run(pass, P.orient, groups.orient[i], ...img.sizes[i]);
        }
    }

    // ---------------------------------------------------------------- views
    const currentView = () => FLOW[state.stage].views[state.view];
    // level toggles apply in Enhancement and Final Image while "All" is selected
    const maskActive = () => { const v = currentView(); return (v.levels && state.enhLevel < 0) || (v.uniform && state.uniformLevel < 0); };
    const excludedBits = () => (maskActive() ? state.levelOn.reduce((b, on, s) => b | (on ? 0 : 1 << s), 0) : 0);
    // uniform level only applies in the Final Image view (its level buttons); all other views foveate by eccentricity
    const uniformLevel = () => (currentView().uniform ? state.uniformLevel : -1);
    function currentMode() {
        const v = currentView();
        if (v.levels) return state.enhLevel >= 0 ? 18 : 16;   // Enhancement: sum, or the contribution of one level
        if (v.fov) return v.fov[state.fovMethod];
        if (!v.stat) return v.mode;
        const stat = v.structure && state.phaseShow === 'structure' ? v.structure : v.stat;
        return state.variant === 'repaired' ? stat.repaired : stat.measured;
    }
    const activeSet = () => (state.analyseFoveated ? img.fovSet : img.orig);
    const structureOn = () => !!currentView().structure && state.phaseShow === 'structure';
    function levelCount(kind) {
        // the phase semantics of level l need level l+1: one level fewer
        return { pyramid: img.nLevels, laplace: img.nLevels - 1, analysis: img.nAnalysis - (structureOn() ? 1 : 0) }[kind] || 1;
    }
    function auxTexture(l = state.level) {
        const v = currentView(), set = activeSet();
        if (v.levels && state.enhLevel >= 0) return img.bands[state.enhLevel];
        if (structureOn()) return img.sem[Math.min(l, img.sem.length - 1)];
        switch (v.level) {
            case 'pyramid': return l === 0 ? (state.analyseFoveated ? img.fov : img.ycc) : set.levels[l];
            case 'laplace': return set.laplace[l];
            case 'analysis': return state.variant === 'repaired' ? img.rep[l] : set.analysis[l];
            default: return set.analysis[0];
        }
    }
    function displayScale(mode = currentMode(), l = state.level) {
        const lap = state.gpuInfo ? state.gpuInfo.lapStds : [];
        const mods = state.gpuInfo ? state.gpuInfo.mods : [];
        const s = lap[Math.min(l, lap.length - 1)] || 0.02;
        switch (mode) {
            case 3: return 20;                                         // difference of the two foveations: +-0.05
            case 7: return 0.15 / s;                                   // Laplacian: +-3.3 std -> full grey range
            // |G + iH| of the steered filters lies in about [0, 1] at every level (99th percentile 0.6 - 0.9), roughly
            // 10x the Laplacian std, so the intensity views map |G + iH| = MAG_MAX to the top of the colour scale
            // Measured and repaired share one scale, so they can be compared: the repaired intensity carries the
            // intensity of coarser levels (up to the cutoff, with their larger modulator) into level l, so the top of
            // the scale is MAG_MAX * modulator of the coarsest level that can reach level l.
            // 19, 20 (phase as structure): the same scale, it fades out where the intensity is low
            case 10: case 12: case 19: case 20: {
                const k = Math.min(Math.max(l, state.gpuInfo ? state.gpuInfo.cutoff : l), mods.length - 1);
                const top = MAG_MAX * Math.max(mods[k] || 0, 1e-9);
                return mode === 10 || mode === 19 ? (mods[l] || 0) / top : 1 / top;   // measured shows |G + iH| (unmodulated)
            }
            case 16: case 18: return 4;                                // noise enhancement around mid-grey
            default: return 1;
        }
    }
    // Range [min, max] that the display maps to the full colour scale, for display mode `mode` at level l
    function legendRange(lg, mode, l) {
        if (!lg.scale) return [lg.min, lg.max];
        const k = displayScale(mode, l);
        if (mode === 3) return [-1 / k, 1 / k];         // diverging map over [-1, 1]
        if (!lg.positive) return [-0.5 / k, 0.5 / k];
        const mod = state.gpuInfo ? state.gpuInfo.mods[l] || 0 : 0;
        return [0, mode === 10 ? mod / k : 1 / k];   // measured view shows |G + iH|; label it as intensity
    }
    // The min label has a fixed width per view (the widest it gets over the view's levels and variants, and the widest
    // seen so far), so the bar does not move while clicking through levels, only when switching to another view.
    const legendMinChars = {};
    function updateLegend() {
        const v = currentView();
        const lg = v.fov && state.fovMethod === 'diff' ? DIFF_LEGEND
            : v.structure && state.phaseShow === 'structure' ? v.structureLegend : v.legend;
        $('legend').hidden = !lg;
        if (!lg) return;
        if (lg.labels) {   // categorical ends (Line / Edge) instead of numbers
            $('legendMin').style.width = '';
            $('legendBar').style.background = gradient(lg.map);
            [$('legendMin').textContent, $('legendMax').textContent] = lg.labels;
            $('legendUnit').textContent = lg.unit;
            return;
        }
        const [min, max] = legendRange(lg, currentMode(), state.level);
        const key = `${state.stage}/${state.view}/${v.fov ? state.fovMethod : ''}`;
        const levels = v.level && img ? levelCount(v.level) : 1;
        const modes = v.stat ? [v.stat.measured, v.stat.repaired] : [currentMode()];
        let chars = legendMinChars[key] || 0;
        for (let l = 0; l < levels; l++) for (const m of modes) chars = Math.max(chars, fmt(legendRange(lg, m, l)[0]).length);
        legendMinChars[key] = chars;
        $('legendMin').style.width = `${chars}ch`;
        $('legendBar').style.background = gradient(lg.map);
        $('legendMin').textContent = fmt(min);
        $('legendMax').textContent = fmt(max);
        $('legendUnit').textContent = lg.unit;
    }
    function updateControls() {
        document.querySelectorAll('.node').forEach((n, i) => n.classList.toggle('active', i === state.stage));
        document.querySelectorAll('.chip').forEach((c) => c.classList.toggle('on', +c.dataset.stage === state.stage && +c.dataset.view === state.view));
        const v = currentView();
        // level buttons in the top-right corner of the image:
        //   level views:  L0 .. L(n-1)                          -> state.level
        //   Enhancement:  L0 .. L3 (contribution of one level), All (sum)               -> state.enhLevel (-1 = all)
        //   Final Image:  L0 .. L4 = the whole image at that level (uniform), All (foveated by eccentricity)
        //                                                                               -> state.uniformLevel (-1 = all)
        //   With All, a column of level toggles drops down under it (#maskBox): which levels' noise is included.
        const bar = $('levelBox');
        const spec = v.levels ? { all: true, n: NOISE_LEVELS, key: 'enhLevel', tip: 'All: the summed enhancement; Lk: the contribution of level k' }
            : v.uniform ? { all: true, n: 5, key: 'uniformLevel', tip: 'All: foveated by eccentricity; Lk: the whole image at pyramid level k (uniform), with its noise' }
            : v.level && img ? { all: false, n: levelCount(v.level), key: 'level', tip: 'pyramid level' } : null;
        bar.hidden = !spec || !img;
        if (spec && img) {
            if (spec.key === 'level') state.level = Math.min(state.level, spec.n - 1);
            const labels = Array.from({ length: spec.n }, (_, i) => `L${i}`).concat(spec.all ? ['All'] : []);
            const value = (k) => (k < spec.n ? k : -1);   // button index -> state value (All, the last one: -1)
            bar.title = spec.tip;
            if (bar.dataset.labels !== labels.join() + spec.key) {
                bar.dataset.labels = labels.join() + spec.key;
                bar.innerHTML = '';
                labels.forEach((text, k) => {
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.textContent = text;
                    btn.addEventListener('click', () => { state[spec.key] = value(k); updateControls(); requestRender(); });
                    bar.appendChild(btn);
                });
            }
            [...bar.children].forEach((btn, k) => btn.classList.toggle('on', value(k) === state[spec.key]));
        }
        const maskBox = $('maskBox');
        maskBox.hidden = !maskActive();
        if (!maskBox.hidden) {
            if (!maskBox.childElementCount) {
                for (let s = 0; s < NOISE_LEVELS; s++) {
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.textContent = `L${s}`;
                    btn.addEventListener('click', () => { state.levelOn[s] = !state.levelOn[s]; updateControls(); requestRender(); });
                    maskBox.appendChild(btn);
                }
            }
            [...maskBox.children].forEach((btn, s) => {
                btn.classList.toggle('on', state.levelOn[s]);
                btn.title = `${state.levelOn[s] ? 'Leave out' : 'Include'} the noise of level ${s}`;
            });
            // the column hangs from the All button (the last one of the bar)
            const allBtn = bar.lastElementChild;
            if (allBtn) maskBox.style.width = `${allBtn.offsetWidth}px`;
        }
        $('variantBox').hidden = !v.stat;
        $('fovBox').hidden = !v.fov;
        $('enhBox').hidden = !v.gain;   // Final Image: Enhancement On / Off (top left)
        document.querySelectorAll('#enhBox button').forEach((b) => b.classList.toggle('on', +b.dataset.enh === +state.enhOn));
        document.querySelectorAll('#fovBox button').forEach((b) => b.classList.toggle('on', b.dataset.fov === state.fovMethod));
        $('gainBox').hidden = !(v.gain || v.amp);   // Noise Amp.: Final Image and Enhancement
        $('keepBox').hidden = !v.keep;
        $('noiseFromRow').hidden = true;   // hidden toggle: Shift + M or ?noise=measured (see below)
        document.querySelectorAll('#noiseFromBox button').forEach((b) => b.classList.toggle('on', b.dataset.from === state.noiseFrom));
        $('phaseBox').hidden = !v.structure;            // Phase view: Angle / Line / Edge
        document.querySelectorAll('#phaseBox button').forEach((b) => b.classList.toggle('on', b.dataset.phase === state.phaseShow));
        // explanatory note under the image for some views
        $('viewNote').hidden = !v.note;
        if (v.note) $('viewNote').textContent = v.note();
        document.querySelectorAll('#variantBox button').forEach((b) => b.classList.toggle('on', b.dataset.variant === state.variant));
        document.querySelectorAll('#analyseBox button').forEach((b) => b.classList.toggle('on', +b.dataset.analyse === +state.analyseFoveated));
        updateLegend();
    }

    // ---------------------------------------------------------------- per frame
    let pending = false;
    function requestRender() {
        if (pending || !img) return;
        pending = true;
        requestAnimationFrame(render);
    }

    let autoClock = 0, lastFrameT = null;   // time [ms] of the automatic eye movement, last animated frame
    function autoGaze(t) {   // slow, smooth eye movement on a lying figure eight (Lissajous 1:2) while the mouse is outside the image
        const a = 2 * Math.PI * (t / 1000) / 20;   // one loop in 20 s
        return [img.h * (0.5 + 0.28 * Math.sin(2 * a)), img.w * (0.5 + 0.36 * Math.sin(a))];
    }

    function render(t) {
        pending = false;
        const t0 = performance.now();
        // automatic eye movement while the mouse is outside the image; its clock only runs while it moves, so after a
        // pause it continues from where it stopped (paused: the gaze stays where it is)
        const animate = !state.frozen && !state.pointerInside && state.autoGaze;
        if (animate) {
            autoClock += lastFrameT === null ? 0 : t - lastFrameT;
            state.gaze = autoGaze(autoClock);
        }
        lastFrameT = animate ? t : null;

        const c = cutoff();
        const set = activeSet();
        const nLap = img.nLevels - 1;
        const r = consts.ratios;
        // noise amp.: Final Image (0 when the enhancement is off) and Enhancement; 1 elsewhere
        const gain = currentView().gain ? (state.enhOn ? state.gain : 0) : currentView().amp ? state.gain : 1;
        device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([
            img.w, img.h, pxPerCm(), consts.sigma0,
            state.gaze[0], state.gaze[1], uniformLevel(), img.maxLayer,
            0, 0, state.distance, currentMode(),
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
            canvas.width, canvas.height, state.showGaze && !currentView().noGaze ? 1 : 0, gain,
            currentMode() === 18 ? state.enhLevel : state.level, displayScale(), 0, c,
        ]));
        device.queue.writeBuffer(composeBuffer, 0, new Float32Array([c, 0, gain, excludedBits()]));   // base: per-pixel blur
        const wf = state.fitWeighted ? 1 : 0;
        device.queue.writeBuffer(fitParamsBuffer, 0, new Float32Array([c, nLap, wf, 0]));
        img.lapParts.forEach((lp, i) => {
            device.queue.writeBuffer(lp.statsU, 0, new Float32Array([i, lp.groups[0], wf, 0]));
            device.queue.writeBuffer(lp.finalizeU, 0, new Float32Array([lp.groups[0] * lp.groups[1], i, wf, 0]));
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        const [w, h] = [img.w, img.h];
        run(pass, P.foveate, img.bg.foveate, w, h);
        run(pass, P.foveatePyr, img.bg.foveatePyr, w, h);
        if (state.analyseFoveated) buildPyramid(pass, img.fovSet, img.fovSet.build);
        // Laplacian std per band and intensity fit
        for (const ls of set.lapStats) {
            runGroups(pass, P.lapStats, ls.stats, ...ls.groups);
            runGroups(pass, P.lapFinalize, ls.finalize, 1, 1);
        }
        runGroups(pass, P.fit, img.bg.fit, 1, 1);
        // repair: levels >= cutoff keep their measured statistics; below, from coarse to fine.
        // Noise from the measured statistics: every level keeps its measured statistics (no extrapolation).
        const fromMeasured = state.noiseFrom === 'meas';
        for (let s = fromMeasured ? 0 : c; s < img.nAnalysis; s++) run(pass, P.repInit, set.repInit[s], ...img.sizes[s]);
        for (let s = fromMeasured ? -1 : c - 1; s >= 0; s--) {
            run(pass, P.repPhi, set.repPhi[s], ...img.sizes[s + 1]);
            run(pass, P.repLevel, set.repLevel[s], ...img.sizes[s]);
        }
        if (structureOn()) {   // Line / Edge view: phase semantics of the shown level
            const l = Math.min(state.level, img.sem.length - 1);
            run(pass, P.sem, set.sem[state.variant][l], ...img.sizes[l]);
        }
        // noise bands of the levels below the cutoff, their std at full resolution, composition
        const randomize = Math.min(...state.keep) < 1 ? 1 : 0;
        for (let s = 0; s < Math.min(c, img.bands.length); s++) {
            device.queue.writeBuffer(img.bandU[s], 0, new Float32Array([h, w, s, randomize, ...state.keep, 0]));
            if (randomize) {   // the random intensity is scaled by the std of the level's intensity_rep
                runGroups(pass, P.lapStats, img.bg.repStd[s].stats, ...img.bg.repStd[s].groups);
                runGroups(pass, P.lapFinalize, img.bg.repStd[s].finalize, 1, 1);
            }
            run(pass, P.band, img.bg.band[s], ...img.sizes[s]);
            runGroups(pass, P.stats, img.bg.bandStats[s].stats, ...img.bg.bandStats[s].groups);
            runGroups(pass, P.finalize, img.bg.bandStats[s].finalize, 1, 1);
        }
        run(pass, P.compose, img.bg.compose, w, h);
        pass.end();

        const display = bindGroup(displayLayout, [[0, paramsBuffer], [1, img.ycc], [2, img.fov], [3, img.maps], [4, img.fovPyr],
            [5, auxTexture()], [6, img.final], [7, bandStdsBuffer], [8, fitBuffer]]);
        const rp = encoder.beginRenderPass({
            colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
        });
        rp.setPipeline(displayPipeline);
        rp.setBindGroup(0, display);
        rp.draw(3);
        rp.end();

        const doReadback = !readbackBusy;
        if (doReadback) {   // fit and Laplacian std for the info line and the legends (arrives a frame later)
            encoder.copyBufferToBuffer(fitBuffer, 0, readback, 0, 48);
            encoder.copyBufferToBuffer(lapStatsBuffer, 0, readback, 48, 64);
            readbackBusy = true;
        }
        device.queue.submit([encoder.finish()]);
        if (doReadback) {
            readback.mapAsync(GPUMapMode.READ).then(() => {
                const v = new Float32Array(readback.getMappedRange().slice(0));
                readback.unmap();
                readbackBusy = false;
                const first = !state.gpuInfo;
                const stats = Array.from({ length: nLap }, (_, i) => [v[12 + 2 * i], v[13 + 2 * i]]);
                state.gpuInfo = { mods: Array.from(v.slice(0, 8)), m: v[8], b: v[9], cutoff: c,
                                  lapStds: stats.map((x) => x[0]), weights: stats.map((x) => x[1]) };
                updateLegend();
                drawFitPlot();
                if (first) requestRender();   // display scales depend on these values
            });
        }
        device.queue.onSubmittedWorkDone().then(() => { $('frame').textContent = `${(performance.now() - t0).toFixed(1)} ms / frame`; });
        if (animate) requestRender();
    }

    // Intensity fit plot (SVG): band std per level (point size = weight in the fit), the fitted line, and the noise
    // amplitude of the synthesized levels (levels below the cutoff).
    function drawFitPlot() {
        const g = state.gpuInfo, el = $('fitPlot');
        if (!g || !el) return;
        const n = g.lapStds.length, c = g.cutoff;
        // drawn at the element's own width (fills the row, text stays the same size); 560 while hidden
        const W = Math.max(320, Math.round(el.clientWidth) || 560), H = 210, L = 46, R = 16, T = 14, B = 34;
        const yMax = Math.max(...g.lapStds, g.m * (n - 0.5) + g.b, 1e-6) * 1.15;
        const X = (lv) => L + (lv + 0.5) / n * (W - L - R);
        const Y = (v) => T + (1 - v / yMax) * (H - T - B);
        const parts = [];
        // levels that get noise
        if (c > 0) parts.push(`<rect x="${X(-0.5)}" y="${T}" width="${X(c - 0.5) - X(-0.5)}" height="${H - T - B}" fill="currentColor" opacity="0.06"/>`,
            `<text x="${(X(-0.5) + X(c - 0.5)) / 2}" y="${T + 12}" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">Synthesized Levels</text>`);
        // axes
        parts.push(`<path d="M${L} ${T} V${H - B} H${W - R}" fill="none" stroke="currentColor" stroke-width="1" opacity="0.6"/>`);
        for (let i = 0; i < n; i++) parts.push(`<text x="${X(i)}" y="${H - B + 16}" text-anchor="middle" font-size="11" fill="currentColor">L${i}</text>`);
        parts.push(`<text x="${L - 6}" y="${Y(0) + 4}" text-anchor="end" font-size="11" fill="currentColor">0</text>`,
            `<text x="${L - 6}" y="${Y(yMax / 1.15) + 4}" text-anchor="end" font-size="11" fill="currentColor">${(yMax / 1.15).toFixed(3)}</text>`,
            `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">Pyramid Level · Std of the Laplacian Band</text>`);
        // fitted line
        // starts where it crosses 0 if it is negative at the left edge (clamping only the start point would change its slope)
        const x1 = n - 0.5, x0 = g.m * -0.5 + g.b < 0 && g.m > 0 ? -g.b / g.m : -0.5;
        parts.push(`<line x1="${X(x0)}" y1="${Y(g.m * x0 + g.b)}" x2="${X(x1)}" y2="${Y(g.m * x1 + g.b)}" stroke="var(--highlight-color, #be0b4b)" stroke-width="1.8"/>`);
        // measured band std; filled = in the fit, radius ~ weight
        for (let i = 0; i < n; i++) {
            const inFit = state.fitWeighted ? g.weights[i] > 0 : i >= c;
            const r = state.fitWeighted ? 2.5 + 4 * g.weights[i] : 5;
            parts.push(inFit
                ? `<circle cx="${X(i)}" cy="${Y(g.lapStds[i])}" r="${r}" fill="var(--gray-dark, #9a8f7c)"/>`
                : `<circle cx="${X(i)}" cy="${Y(g.lapStds[i])}" r="3.5" fill="none" stroke="currentColor" opacity="0.5"/>`);
        }
        // noise amplitude of the synthesized levels
        for (let i = 0; i < c; i++) {
            const x = X(i), y = Y(g.mods[i]);
            parts.push(`<path d="M${x} ${y - 5} L${x + 5} ${y} L${x} ${y + 5} L${x - 5} ${y} Z" fill="var(--highlight-color, #be0b4b)"/>`);
        }
        el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="intensity fit">${parts.join('')}</svg>` +
            `<div class="pam-fit-caption">${state.fitWeighted
                ? 'Weighted fit: every band is measured only where its level is fully kept (mask = 1); point size = that fraction of the image (weight in the fit).'
                : 'Paper fit: line through the bands from the cutoff level on (filled points).'} ` +
            `Diamonds: noise amplitude of the synthesized levels.</div>`;
    }

    function resizeCanvas() {
        if (!img) return;
        const dpr = window.devicePixelRatio || 1;
        const box = canvas.parentElement;
        const cssWidth = document.fullscreenElement === box
            ? Math.min(window.innerWidth, window.innerHeight * img.w / img.h)   // full screen: fit the screen
            : box.clientWidth;
        state.cssWidth = cssWidth;
        updateSetupInfo();
        canvas.style.width = `${cssWidth}px`;
        canvas.style.height = `${cssWidth * img.h / img.w}px`;
        canvas.width = Math.round(cssWidth * dpr);
        canvas.height = Math.round(cssWidth * img.h / img.w * dpr);
        requestRender();
    }

    // ---------------------------------------------------------------- screen setup and full screen
    // the slider shows the current width (it changes with the window, zoom and full screen)
    function updateSetupInfo() {
        if (!img || !state.cssWidth) return;
        const cm = imageWidthCm();
        $('imageCm').value = cm.toFixed(1);
        $('imageCmVal').textContent = `${cm.toFixed(1)} cm`;
        if (typeof updateFills === 'function') updateFills();
    }
    try {   // remember the calibration (this browser only)
        const saved = +localStorage.getItem('pam-cm-per-device-px');
        if (saved > 0) state.cmPerDevicePx = saved;
    } catch (e) { /* storage unavailable: default */ }
    $('imageCm').addEventListener('input', () => {
        const cm = +$('imageCm').value;
        if (!(cm > 0) || !state.cssWidth) return;
        state.cmPerDevicePx = cm / shownDevicePx();
        $('imageCmVal').textContent = `${cm.toFixed(1)} cm`;
        try { localStorage.setItem('pam-cm-per-device-px', String(state.cmPerDevicePx)); } catch (e) { /* ignore */ }
        requestRender();
    });
    // full-screen button in the corner of the image, as on a video
    const FS_ICON = {
        // 16:9 frame corners, pointing out (enter) or in (exit)
        enter: '<svg viewBox="0 0 32 18" width="27.2" height="15.3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 6V1h6M25 1h6v5M31 12v5h-6M7 17H1v-5"/></svg>',
        exit: '<svg viewBox="0 0 32 18" width="27.2" height="15.3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 1v5H1M31 6h-6V1M25 17v-5h6M1 12h6v5"/></svg>',
    };
    const fsButton = $('fullscreen');
    fsButton.innerHTML = FS_ICON.enter;
    fsButton.addEventListener('click', () => {
        const box = canvas.parentElement;
        if (document.fullscreenElement) document.exitFullscreen();
        else if (box.requestFullscreen) box.requestFullscreen().catch((e) => status(`Full screen not available: ${e.message}`));
    });
    document.addEventListener('fullscreenchange', () => {
        const on = document.fullscreenElement === canvas.parentElement;
        fsButton.innerHTML = on ? FS_ICON.exit : FS_ICON.enter;
        fsButton.setAttribute('aria-label', on ? 'Leave full screen' : 'Full screen');
        resizeCanvas();
    });

    // ---------------------------------------------------------------- UI
    FLOW.forEach((stage, si) => {
        const node = document.createElement('div');
        node.className = 'node';
        node.innerHTML = `<div class="node-title">${stage.title}</div><div class="chips"></div>`;
        stage.views.forEach((v, vi) => {
            const chip = document.createElement('button');
            chip.className = 'chip';
            chip.textContent = v.name;
            chip.dataset.stage = si;
            chip.dataset.view = vi;
            chip.addEventListener('click', () => {
                // entering the Image Pyramids or Statistics stage: show what a renderer has, the foveated image
                if (si !== state.stage && stage.toFoveated && !state.analyseFoveated) { state.analyseFoveated = true; state.gpuInfo = null; }
                state.stage = si; state.view = vi; updateControls(); requestRender();
            });
            node.querySelector('.chips').appendChild(chip);
        });
        $('flow').appendChild(node);
    });
    // Flowchart layout from the text, not from the window size: 6 x 1, else 3 x 2, else 2 x 3, else 1 x 6 -- the first in which
    // every chip label fits on one line with FLOW_MARGIN px to spare. Re-checked on resize and once the fonts are loaded.
    const FLOW_MARGIN = 6;
    // (measured on the text itself: buttons do not reliably report their overflow in scrollWidth)
    const range = document.createRange();
    function chipFits(c) {
        range.selectNodeContents(c);
        const st = getComputedStyle(c);
        const needed = range.getBoundingClientRect().width + parseFloat(st.paddingLeft) + parseFloat(st.paddingRight);
        return needed + FLOW_MARGIN <= c.getBoundingClientRect().width;
    }
    function layoutFlow() {
        const flow = $('flow'), chips = [...flow.querySelectorAll('.chip')];
        for (const cols of [6, 3, 2, 1]) {
            flow.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
            flow.dataset.cols = cols;
            if (chips.every(chipFits)) break;
        }
    }
    let flowWidth = -1;   // react to width changes only (the reflow itself changes the container's height)
    new ResizeObserver((entries) => {
        const w = Math.round(entries[0].contentRect.width);
        if (w !== flowWidth) { flowWidth = w; layoutFlow(); }
    }).observe($('flow').parentElement);
    if (document.fonts) document.fonts.ready.then(layoutFlow);
    layoutFlow();

    const gazeFromEvent = (e) => {
        const rect = canvas.getBoundingClientRect();
        return [(e.clientY - rect.top) / rect.height * img.h, (e.clientX - rect.left) / rect.width * img.w];
    };
    canvas.addEventListener('pointerenter', () => { state.pointerInside = true; });
    canvas.addEventListener('pointerleave', () => { state.pointerInside = false; requestRender(); });
    canvas.addEventListener('pointermove', (e) => {
        state.pointerInside = true;
        if (!img || state.frozen) return;
        state.gaze = gazeFromEvent(e);
        requestRender();
    });
    canvas.addEventListener('click', (e) => {   // click: freeze / release the gaze
        if (!img) return;
        state.frozen = !state.frozen;
        state.gaze = gazeFromEvent(e);
        requestRender();
    });

    // Noise from the measured instead of the extrapolated statistics: no visible control; Shift + M toggles it,
    // ?noise=measured in the URL starts with it (the #noiseFromRow buttons stay in the page, hidden)
    if (new URLSearchParams(location.search).get('noise') === 'measured') state.noiseFrom = 'meas';
    document.addEventListener('keydown', (e) => {
        if (!e.shiftKey || e.ctrlKey || e.altKey || e.metaKey || e.key.toLowerCase() !== 'm') return;
        if (e.target.closest && e.target.closest('input, select, textarea')) return;
        state.noiseFrom = state.noiseFrom === 'meas' ? 'rep' : 'meas';
        status(`Noise from the ${state.noiseFrom === 'meas' ? 'measured' : 'extrapolated'} statistics`);
        updateControls();
        requestRender();
    });
    document.querySelectorAll('#noiseFromBox button').forEach((b) => b.addEventListener('click', () => {
        state.noiseFrom = b.dataset.from; updateControls(); requestRender();
    }));
    document.querySelectorAll('#enhBox button').forEach((b) => b.addEventListener('click', () => {
        state.enhOn = b.dataset.enh === '1'; updateControls(); requestRender();
    }));
    document.querySelectorAll('#fovBox button').forEach((b) => b.addEventListener('click', () => {
        state.fovMethod = b.dataset.fov; updateControls(); requestRender();
    }));
    document.querySelectorAll('#variantBox button').forEach((b) => b.addEventListener('click', () => {
        state.variant = b.dataset.variant; updateControls(); requestRender();
    }));
    document.querySelectorAll('#phaseBox button').forEach((b) => b.addEventListener('click', () => {
        state.phaseShow = b.dataset.phase; updateControls(); requestRender();
    }));
    // fit plot: hidden until opened with the chart button next to the Intensity Fit toggle
    $('fitPlotBtn').addEventListener('click', () => {
        const open = $('fitPlot').hidden;
        $('fitPlot').hidden = !open;
        $('fitPlotBtn').classList.toggle('on', open);
        $('fitPlotBtn').setAttribute('aria-expanded', String(open));
        $('fitPlotBtn').title = open ? 'Hide the intensity fit plot' : 'Show the intensity fit plot';
        if (open) drawFitPlot();
    });
    document.querySelectorAll('#fitBox button').forEach((b) => b.addEventListener('click', () => {
        state.fitWeighted = b.dataset.fit === 'weighted';
        document.querySelectorAll('#fitBox button').forEach((x) => x.classList.toggle('on', x === b));
        requestRender();
    }));
    document.querySelectorAll('#analyseBox button').forEach((b) => b.addEventListener('click', () => {
        state.analyseFoveated = b.dataset.analyse === '1'; state.gpuInfo = null; updateControls(); requestRender();
    }));
    $('distance').addEventListener('input', (e) => {
        state.distance = +e.target.value;
        $('distanceVal').textContent = `${state.distance} cm`;
        requestRender();
    });
    $('gain').addEventListener('input', (e) => {
        state.gain = +e.target.value;
        $('gainVal').textContent = state.gain.toFixed(2);
        requestRender();
    });
    ['keepI', 'keepD', 'keepP'].forEach((id, k) => $(id).addEventListener('input', (e) => {
        state.keep[k] = +e.target.value;
        $(`${id}Val`).textContent = state.keep[k].toFixed(2);
        requestRender();
    }));
    // Fill sliders: a range input drawn as a button that fills from the left (CSS --p = position in %), with its
    // label on top; a second, light copy of the label is clipped to the filled part so it stays readable there.
    const fills = [...document.querySelectorAll('.pam-fill')].map((box) => {
        const input = box.querySelector('input[type=range]'), dark = box.querySelector('.pam-fill-text');
        const light = document.createElement('span');
        light.className = 'pam-fill-text pam-fill-light';
        light.setAttribute('aria-hidden', 'true');
        box.appendChild(light);
        const update = () => {
            const p = (input.value - input.min) / (input.max - input.min) * 100;
            box.style.setProperty('--p', `${p}%`);
            light.textContent = dark.textContent;
        };
        input.addEventListener('input', update);   // registered after the value labels are updated
        return update;
    });
    var updateFills = () => fills.forEach((f) => f());   // var: also callable from updateSetupInfo
    // icon buttons next to the full-screen button: pause / resume the automatic eye movement, hide / show the gaze marker
    // Every icon's viewBox is cropped to its drawn shape (stroke included) and drawn at the same scale (0.85 px per
    // unit, as the full-screen icon), so equal button padding gives equal visible gaps between the icons. The two
    // states of a button share one box, so toggling does not move the others.
    const ICON = (box, path) => {
        const [, , w, h] = box;
        return `<svg viewBox="${box.join(' ')}" width="${w * 0.85}" height="${h * 0.85}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${path}</svg>`;
    };
    const PLAY_BOX = [6, 3.5, 12, 17], EYE_BOX = [1, 4, 22, 16], IMAGE_BOX = [1, 3, 22, 18];
    const SCENE_ICON = ICON(IMAGE_BOX, '<rect x="2" y="4" width="20" height="16" rx="2"/><circle cx="8" cy="9.5" r="1.8"/><path d="M3 18l6-6 4 4 3-3 5 5"/>');
    const GAZE_ICON = {
        pause: ICON(PLAY_BOX, '<path d="M7 5v14M17 5v14"/>'),
        play: ICON(PLAY_BOX, '<path d="M7 4.5v15L17 12z"/>'),
        eye: ICON(EYE_BOX, '<path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>'),
        eyeOff: ICON(EYE_BOX, '<path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z"/><circle cx="12" cy="12" r="3"/><path d="M5 5l14 14"/>'),
    };
    function updateGazeButtons() {
        const a = $('autoGazeBtn'), g = $('showGazeBtn');
        a.innerHTML = state.autoGaze ? GAZE_ICON.pause : GAZE_ICON.play;
        a.title = state.autoGaze ? 'Pause the automatic eye movement (it runs while the mouse is not over the image)'
                                 : 'Resume the automatic eye movement (it runs while the mouse is not over the image)';
        g.innerHTML = state.showGaze ? GAZE_ICON.eye : GAZE_ICON.eyeOff;
        g.title = state.showGaze ? 'Hide the gaze marker' : 'Show the gaze marker';
        a.setAttribute('aria-label', a.title);
        g.setAttribute('aria-label', g.title);
    }
    $('autoGazeBtn').addEventListener('click', () => { state.autoGaze = !state.autoGaze; updateGazeButtons(); requestRender(); });
    $('showGazeBtn').addEventListener('click', () => { state.showGaze = !state.showGaze; updateGazeButtons(); requestRender(); });
    updateGazeButtons();
    // scene button: cycles through the scenes listed in the (hidden) #image select of the page
    const sceneSelect = $('image'), sceneBtn = $('sceneBtn');
    let sceneLoading = false;
    sceneBtn.innerHTML = SCENE_ICON;
    // info icon: opens a short explanation above the icons; closes on a second click, a click elsewhere or Esc
    const infoBtn = $('infoBtn'), infoPop = $('infoPop');
    infoBtn.innerHTML = ICON([2, 2, 20, 20], '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.6v.01"/>');
    const setInfo = (open) => {
        infoPop.hidden = !open;
        infoBtn.setAttribute('aria-expanded', String(open));
        infoBtn.title = open ? 'Close the explanation' : 'How to use the demo';
        infoBtn.setAttribute('aria-label', infoBtn.title);
    };
    setInfo(false);
    infoBtn.addEventListener('click', (e) => { e.stopPropagation(); setInfo(infoPop.hidden); });
    document.addEventListener('click', (e) => { if (!infoPop.hidden && !infoPop.contains(e.target)) setInfo(false); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !infoPop.hidden) setInfo(false); });
    function updateSceneButton() {
        const opts = sceneSelect.options, i = sceneSelect.selectedIndex;
        sceneBtn.title = `Scene: ${opts[i].text} (${i + 1} / ${opts.length}). Click for the next one: ${opts[(i + 1) % opts.length].text}`;
        sceneBtn.setAttribute('aria-label', sceneBtn.title);
    }
    sceneBtn.addEventListener('click', async () => {
        if (sceneLoading) return;   // one scene at a time
        sceneLoading = true;
        sceneSelect.selectedIndex = (sceneSelect.selectedIndex + 1) % sceneSelect.options.length;
        updateSceneButton();
        try { await loadImage(sceneSelect.value); } catch (err) { status(err.message, true); }
        sceneLoading = false;
    });
    updateSceneButton();
    window.addEventListener('resize', resizeCanvas);

    updateControls();
    updateFills();
    await loadImage($('image').value);
    started = true;
}

main().catch((err) => {
    const msg = `Error: ${err.message}`;
    if (started || !showFallback(msg)) status(msg, true);
});
