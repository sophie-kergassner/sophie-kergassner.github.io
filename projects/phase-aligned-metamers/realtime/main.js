// Phase-aligned metamers: WebGPU version of the full pipeline.
//   per image: YCbCr, blur stack, Gaussian/Laplacian pyramid and analysis of the original image
//   per frame: eccentricity -> level -> sigma, foveation, [pyramid + analysis of the foveated image],
//              Laplacian std + intensity fit, repair, noise synthesis, composition
// The host logic mirrors web/test_webgpu.py, which checks every shader against the Python package.

const SCREEN_WIDTH_CM = 120;   // physical display width [cm]
const SCREEN_WIDTH_PX = 3840;  // horizontal display resolution [px]
const MAX_PYR_DEPTH = 5;       // as in example.ipynb: cutoff <= 4
const ANALYSIS_LEVELS = MAX_PYR_DEPTH + 1;   // levels 0..5 (cutoff + 1 extra level for the phase repair)
const NOISE_LEVELS = MAX_PYR_DEPTH - 1;      // noise bands for levels 0..3 (s < cutoff <= 4)
const MAG_MAX = 0.8;                          // |G + iH| shown at the top of the colour scale (intensity views)
const MIN_LEVEL_SIZE = 8;      // Worm.MIN_LEVEL_SIZE

// Pipeline stages (flowchart). Each view: display mode (display.wgsl), level = which levels the slider offers,
// stat = statistic with the measured / repaired switch, legend = colour scale.
const FLOW = [
    { title: 'Viewing geometry', views: [
        { name: 'Eccentricity', mode: 4, legend: { map: 'inferno', min: 0, max: 40, unit: 'deg' } },
        { name: 'Level', mode: 5, legend: { map: 'inferno', min: 0, max: 4, unit: 'pyramid level' } },
        { name: 'Blur σ', mode: 6, legend: { map: 'inferno', min: 0, max: 10, unit: 'px' } },
    ] },
    { title: 'Foveation', views: [
        { name: 'Original', mode: 0, noGaze: true },
        { name: 'Per-pixel blur', mode: 1 },
        { name: 'Pyramid', mode: 2 },
        { name: 'Difference', mode: 3, legend: { map: 'diverging', scale: true, unit: 'Y (blur − pyramid)' } },
    ] },
    { title: 'Pyramid', views: [
        { name: 'Image', mode: 11, level: 'pyramid' },
        { name: 'Laplacian band', mode: 7, level: 'laplace', legend: { map: 'gray', scale: true, unit: 'Y' } },
    ] },
    { title: 'Statistics', views: [
        { name: 'Orientation', stat: { measured: 8, repaired: 13 }, level: 'analysis', legend: { map: 'cyclic', min: -90, max: 90, unit: 'deg' } },
        { name: 'Phase', stat: { measured: 9, repaired: 14 }, level: 'analysis', legend: { map: 'cyclic', min: -180, max: 180, unit: 'deg' } },
        { name: 'Intensity', stat: { measured: 10, repaired: 12 }, level: 'analysis', legend: { map: 'inferno', scale: true, positive: true, unit: '|G + iH| · modulator' } },
    ] },
    { title: 'Noise', views: [
        { name: 'Band', mode: 17, level: 'noise', legend: { map: 'gray', scale: true, unit: 'std of the band' } },
        { name: 'Enhancement', mode: 16, levels: true, legend: { map: 'gray', scale: true, unit: 'Y' } },
    ] },
    { title: 'Result', views: [
        { name: 'Final image', mode: 15, gain: true },
    ] },
];

const $ = (id) => document.getElementById(id);
// The demo can be embedded in another page: the element with [data-pam-demo] may set data-base, the path
// (relative to the page) of the folder that contains shaders/ and constants.json.
const BASE = (document.querySelector('[data-pam-demo]') || { dataset: {} }).dataset.base || '';
let errorShown = false;
const status = (msg, isError = false) => {
    if (errorShown && !isError) return;   // keep the first error visible
    errorShown = errorShown || isError;
    $('status').textContent = msg;
    $('status').classList.toggle('error', isError);
};

const state = {
    stage: 5, view: 0,      // selected flowchart node and view
    variant: 'measured',    // statistics: measured / repaired
    level: 0,               // pyramid level of the level views
    gaze: null,             // [row, col] in image pixels
    frozen: false,          // click on the image freezes the gaze
    pointerInside: false,
    autoGaze: true,         // slow automatic eye movement while the mouse is outside the image
    motion: false,          // moving input: the image is translated every frame (temporal stability test)
    distance: 50,           // viewing distance [cm]
    uniformLevel: -1,       // < 0: eccentricity-dependent foveation
    showGaze: true,
    analyseFoveated: true,  // statistics measured on the foveated image (false: on the full-res original, as in the paper)
    fitWeighted: true,      // intensity fit: weighted over all bands (web version) or from the cutoff on (paper)
    enhLevel: -1,           // Enhancement view: -1 = sum of all levels, s = contribution of level s only
    gain: 1,                // noise gain of the "Final image" view (1 = as computed, 0 = no noise)
    gpuInfo: null,          // last read-back of the intensity fit and the Laplacian std (for info line and legends)
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
const cyclic = (t) => [0, 1 / 3, 2 / 3].map((o) => 0.5 + 0.45 * Math.cos(2 * Math.PI * (t + o)));
function diverging(v) {
    const mix = (a, b, f) => a.map((x, i) => x + (b[i] - x) * f);
    return v < 0 ? mix([1, 1, 1], [0.13, 0.40, 0.67], -v) : mix([1, 1, 1], [0.70, 0.09, 0.17], v);
}
function gradient(map) {
    const stops = [];
    for (let i = 0; i <= 16; i++) {
        const t = i / 16;
        const rgb = map === 'inferno' ? inferno(t) : map === 'cyclic' ? cyclic(t) : map === 'diverging' ? diverging(2 * t - 1) : [t, t, t];
        stops.push(`rgb(${rgb.map((x) => Math.round(255 * x)).join(',')}) ${(100 * t).toFixed(1)}%`);
    }
    return `linear-gradient(to right, ${stops.join(', ')})`;
}
const fmt = (v) => (Math.abs(v) >= 100 || v === 0 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toPrecision(3) : v.toPrecision(2));

async function main() {
    if (!navigator.gpu) {
        status('This browser does not support WebGPU. Use a current Chrome, Edge or Safari (or Firefox on Windows).', true);
        return;
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) { status('No WebGPU adapter found.', true); return; }
    const device = await adapter.requestDevice();
    device.lost.then((info) => status(`GPU device lost: ${info.message}`, true));
    device.addEventListener('uncapturederror', (e) => status(`WebGPU error: ${e.error.message}`, true));

    const names = ['common', 'sampling', 'ycbcr', 'blur', 'foveate', 'foveate_pyramid', 'pyramid_down', 'laplace',
                   'analysis_filters', 'analysis_orient', 'lap_stats', 'lap_finalize', 'fit', 'repair_common', 'repair_init', 'repair_phi', 'repair_level',
                   'noise_band', 'noise_stats', 'noise_finalize', 'compose', 'display', 'shift'];
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
        shift: computePipeline(code('shift'), [uniform(0), tex(1), storage(2, 'rgba32float')]),
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
        band: computePipeline(code('noise_band'), [uniform(0), roStorage(1), tex(2), storage(3, 'r32float')]),
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
    const kernelBuffer = fixedBuffer(consts.pyr_kernel, GPUBufferUsage.STORAGE);
    const filterBuffer = fixedBuffer(consts.filters, GPUBufferUsage.STORAGE);
    const bandStdsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE });
    const lapStatsBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });   // (std, weight) per band
    const fitBuffer = device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
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
    function eccentricity(row, col) {
        const ppc = SCREEN_WIDTH_PX / SCREEN_WIDTH_CM;
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
        if (state.uniformLevel >= 0) {
            maxLevel = state.uniformLevel;
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
        img.base = make(w, h, 'rgba32float');   // YCbCr of the loaded image
        img.ycc = make(w, h, 'rgba32float');    // input of the pipeline: the base image, shifted when the image moves
        // kept for the moving input, which rebuilds the blur stack every frame
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

        // ---- input: base image, shifted by img.offset (moving input); blur stack of the input (bind groups)
        img.offset = [0, 0];
        img.shiftU = uniformBuffer(16);
        buffers.push(img.shiftU);
        img.bg = { shift: bg(P.shift, [img.shiftU, img.base, img.ycc]), blur: [] };
        let cur = img.ycc;
        for (let k = 1; k <= maxLayer; k++) {
            const sigma = Math.sqrt(k * k - (k - 1) * (k - 1));
            const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
            const ab = ub.getMappedRange();
            new Float32Array(ab, 0, 1)[0] = sigma;
            new Int32Array(ab, 4, 1)[0] = Math.trunc(2 * sigma + 0.5);
            ub.unmap();
            buffers.push(ub);
            const next = cur === ping ? pong : ping;
            img.bg.blur.push([bg(P.blurV, [ub, cur, tmp]),
                bg(P.blurH, [ub, tmp, next, img.stack.createView({ dimension: '2d', baseArrayLayer: k, arrayLayerCount: 1 })])]);
            cur = next;
        }

        // ---- per-image computation: YCbCr, input, blur stack, pyramid and analysis of the original
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        run(pass, P.ycbcr, bg(P.ycbcr, [img.src, img.base]), w, h);
        buildInput(pass);
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // ---- per-frame resources and bind groups
        img.rep = img.sizes.slice(0, nAnalysis).map((s) => make(...s, 'rgba32float'));
        img.phi = img.sizes.slice(0, nAnalysis).map((s) => make(...s, 'rgba32float'));
        img.bands = img.sizes.slice(0, NOISE_LEVELS).map((s) => make(...s, 'r32float'));
        const partial = (gx, gy) => {
            const b = device.createBuffer({ size: gx * gy * 8, usage: GPUBufferUsage.STORAGE });
            buffers.push(b);
            return b;
        };
        const groups16 = ([sw, sh]) => [Math.ceil(sw / 16), Math.ceil(sh / 16)];

        const lvViews = [0, 1, 2, 3, 4, 5].map((i) => img.orig.levels[Math.min(i, nLevels - 1)]);
        Object.assign(img.bg, {
            foveate: bg(P.foveate, [paramsBuffer, img.ycc, img.stack.createView({ dimension: '2d-array' }), img.maps, img.fov]),
            foveatePyr: bg(P.foveatePyr, [paramsBuffer, ...lvViews, img.fovPyr]),
            fit: bg(P.fit, [fitParamsBuffer, lapStatsBuffer, fitBuffer]),
            band: img.bands.map((b, s) => bg(P.band, [fixedUniform([h, w, 0, 0]), filterBuffer, img.rep[s], b])),
            bandStats: img.bands.map((b, s) => {
                const [gx, gy] = groups16([w, h]), part = partial(gx, gy);
                return { groups: [gx, gy],
                         stats: bg(P.stats, [fixedUniform([2 ** s, gx, 0, 0]), b, img.ycc, part]),
                         finalize: bg(P.finalize, [fixedUniform([gx * gy, w * h, s, 0]), part, bandStdsBuffer]) };
            }),
            compose: bg(P.compose, [paramsBuffer, composeBuffer, img.maps, img.fov, img.fovPyr,
                ...[0, 1, 2, 3].map((s) => img.bands[Math.min(s, img.bands.length - 1)]), bandStdsBuffer, img.final, fitBuffer]),
        });
        // bind groups that depend on the analysed set
        const lapParts = img.sizes.slice(0, nLevels - 1).map((s) => {
            const [gx, gy] = groups16(s);
            const b = device.createBuffer({ size: gx * gy * 16, usage: GPUBufferUsage.STORAGE });
            buffers.push(b);
            return { groups: [gx, gy], buf: b, statsU: uniformBuffer(16), finalizeU: uniformBuffer(16) };
        });
        img.lapParts = lapParts;
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
        }

        state.gaze = [h / 2, w / 2];
        state.gpuInfo = null;
        updateControls();
        resizeCanvas();
        status(`${w} × ${h} px · per-image precomputation in ${(performance.now() - t0).toFixed(0)} ms`);
        requestRender();
    }

    // Input of the pipeline: the base image shifted by img.offset, its blur stack (layer k: sigma = k, built
    // incrementally) and the pyramid and analysis of the original. Once per image, or every frame when the image moves.
    function buildInput(pass) {
        device.queue.writeBuffer(img.shiftU, 0, new Float32Array([img.offset[0], img.offset[1], 0, 0]));
        run(pass, P.shift, img.bg.shift, img.w, img.h);
        for (const [v, hz] of img.bg.blur) {
            run(pass, P.blurV, v, img.w, img.h);
            run(pass, P.blurH, hz, img.w, img.h);
        }
        buildPyramid(pass, img.orig, img.orig.build);
    }
    // moving input: a slow circle of 48 px radius, one round in 12 s (about 25 px/s), in whole pixels
    function motionOffset(t) {
        if (!state.motion) return [0, 0];
        const a = 2 * Math.PI * (t / 1000) / 12;
        return [Math.round(48 * Math.cos(a)) - 48, Math.round(48 * Math.sin(a))];
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
    function currentMode() {
        const v = currentView();
        if (v.levels) return state.enhLevel >= 0 ? 18 : 16;   // Enhancement: sum, or the contribution of one level
        if (!v.stat) return v.mode;
        return state.variant === 'repaired' ? v.stat.repaired : v.stat.measured;
    }
    const activeSet = () => (state.analyseFoveated ? img.fovSet : img.orig);
    function levelCount(kind) {
        return { pyramid: img.nLevels, laplace: img.nLevels - 1, noise: NOISE_LEVELS, analysis: img.nAnalysis }[kind] || 1;
    }
    function auxTexture() {
        const v = currentView(), set = activeSet(), l = state.level;
        if (v.levels && state.enhLevel >= 0) return img.bands[state.enhLevel];
        switch (v.level) {
            case 'pyramid': return l === 0 ? (state.analyseFoveated ? img.fov : img.ycc) : set.levels[l];
            case 'laplace': return set.laplace[l];
            case 'noise': return img.bands[l];
            case 'analysis': return state.variant === 'repaired' ? img.rep[l] : set.analysis[l];
            default: return set.analysis[0];
        }
    }
    function displayScale() {
        const mode = currentMode(), l = state.level;
        const lap = state.gpuInfo ? state.gpuInfo.lapStds : [];
        const mods = state.gpuInfo ? state.gpuInfo.mods : [];
        const s = lap[Math.min(l, lap.length - 1)] || 0.02;
        switch (mode) {
            case 3: return 20;                                         // difference of the two foveations
            case 7: return 0.15 / s;                                   // Laplacian: +-3.3 std -> full grey range
            // |G + iH| of the steered filters lies in about [0, 1] at every level (99th percentile 0.6 - 0.9), roughly
            // 10x the Laplacian std, so the intensity views map |G + iH| = MAG_MAX to the top of the colour scale
            // Measured and repaired share one scale, so they can be compared: the repaired intensity carries the
            // intensity of coarser levels (up to the cutoff, with their larger modulator) into level l, so the top of
            // the scale is MAG_MAX * modulator of the coarsest level that can reach level l.
            case 10: case 12: {
                const k = Math.min(Math.max(l, state.gpuInfo ? state.gpuInfo.cutoff : l), mods.length - 1);
                const top = MAG_MAX * Math.max(mods[k] || 0, 1e-9);
                return mode === 10 ? (mods[l] || 0) / top : 1 / top;   // measured view shows |G + iH| (unmodulated)
            }
            case 16: case 18: return 4;                                // noise enhancement around mid-grey
            case 17: return 0.15;                                      // noise band, normalized by its std
            default: return 1;
        }
    }
    function updateLegend() {
        const v = currentView(), lg = v.legend;
        $('legend').hidden = !lg;
        if (!lg) return;
        let min = lg.min, max = lg.max;
        if (lg.scale) {   // range that the display maps to the full colour scale
            const k = displayScale(), mode = currentMode();
            if (mode === 3) { min = -1 / k; max = 1 / k; }
            else if (lg.positive) {
                min = 0;
                const mod = state.gpuInfo ? state.gpuInfo.mods[state.level] || 0 : 0;
                max = mode === 10 ? mod / k : 1 / k;    // measured view shows |G + iH|; label it as intensity
            } else { min = -0.5 / k; max = 0.5 / k; }
        }
        $('legendBar').style.background = gradient(lg.map);
        $('legendMin').textContent = fmt(min);
        $('legendMax').textContent = fmt(max);
        $('legendUnit').textContent = lg.unit;
    }
    function updateControls() {
        document.querySelectorAll('.node').forEach((n, i) => n.classList.toggle('active', i === state.stage));
        document.querySelectorAll('.chip').forEach((c) => c.classList.toggle('on', +c.dataset.stage === state.stage && +c.dataset.view === state.view));
        const v = currentView();
        $('levelBox').hidden = !v.level;
        if (v.level && img) {
            const n = levelCount(v.level);
            state.level = Math.min(state.level, n - 1);
            $('level').max = n - 1;
            $('level').value = state.level;
            $('levelVal').textContent = `L${state.level}`;
        }
        $('variantBox').hidden = !v.stat;
        $('gainBox').hidden = !v.gain;
        $('enhLevels').hidden = !v.levels;
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

    function autoGaze(t) {   // slow, smooth eye movement on a lying figure eight (Lissajous 1:2) while the mouse is outside the image
        const a = 2 * Math.PI * (t / 1000) / 20;   // one loop in 20 s
        return [img.h * (0.5 + 0.28 * Math.sin(2 * a)), img.w * (0.5 + 0.36 * Math.sin(a))];
    }

    function render(t) {
        pending = false;
        const t0 = performance.now();
        const animate = (!state.frozen && !state.pointerInside && state.autoGaze) || state.motion;
        if (!state.frozen && !state.pointerInside) state.gaze = state.autoGaze ? autoGaze(t) : [img.h / 2, img.w / 2];

        const c = cutoff();
        const set = activeSet();
        const nLap = img.nLevels - 1;
        const r = consts.ratios;
        device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([
            img.w, img.h, SCREEN_WIDTH_PX / SCREEN_WIDTH_CM, consts.sigma0,
            state.gaze[0], state.gaze[1], state.uniformLevel, img.maxLayer,
            0, 0, state.distance, currentMode(),
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
            canvas.width, canvas.height, state.showGaze && !currentView().noGaze ? 1 : 0, 0,
            currentMode() === 18 ? state.enhLevel : state.level, displayScale(), 0, c,
        ]));
        device.queue.writeBuffer(composeBuffer, 0, new Float32Array([c, 0, currentView().gain ? state.gain : 1, 0]));   // base: per-pixel blur
        const wf = state.fitWeighted ? 1 : 0;
        device.queue.writeBuffer(fitParamsBuffer, 0, new Float32Array([c, nLap, wf, 0]));
        img.lapParts.forEach((lp, i) => {
            device.queue.writeBuffer(lp.statsU, 0, new Float32Array([i, lp.groups[0], wf, 0]));
            device.queue.writeBuffer(lp.finalizeU, 0, new Float32Array([lp.groups[0] * lp.groups[1], i, wf, 0]));
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        const [w, h] = [img.w, img.h];
        const off = motionOffset(t);
        if (off[0] !== img.offset[0] || off[1] !== img.offset[1]) {   // moved (or motion switched off): new input
            img.offset = off;
            buildInput(pass);
        }
        run(pass, P.foveate, img.bg.foveate, w, h);
        run(pass, P.foveatePyr, img.bg.foveatePyr, w, h);
        if (state.analyseFoveated) buildPyramid(pass, img.fovSet, img.fovSet.build);
        // Laplacian std per band and intensity fit
        for (const ls of set.lapStats) {
            runGroups(pass, P.lapStats, ls.stats, ...ls.groups);
            runGroups(pass, P.lapFinalize, ls.finalize, 1, 1);
        }
        runGroups(pass, P.fit, img.bg.fit, 1, 1);
        // repair: levels >= cutoff keep their measured statistics; below, from coarse to fine
        for (let s = c; s < img.nAnalysis; s++) run(pass, P.repInit, set.repInit[s], ...img.sizes[s]);
        for (let s = c - 1; s >= 0; s--) {
            run(pass, P.repPhi, set.repPhi[s], ...img.sizes[s + 1]);
            run(pass, P.repLevel, set.repLevel[s], ...img.sizes[s]);
        }
        // noise bands of the levels below the cutoff, their std at full resolution, composition
        for (let s = 0; s < Math.min(c, img.bands.length); s++) {
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
        if (currentView().levels) renderThumbs(encoder, c);

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
                $('info').textContent = `analysis of the ${state.analyseFoveated ? 'foveated' : 'original'} image · cutoff L${c} · ` +
                    `${state.fitWeighted ? 'weighted' : 'paper'} fit: std = ${v[8].toFixed(4)} · level + ${v[9].toFixed(4)}`;
                updateLegend();
                drawFitPlot();
                if (first) requestRender();   // display scales depend on these values
            });
        }
        device.queue.onSubmittedWorkDone().then(() => { $('frame').textContent = `${(performance.now() - t0).toFixed(1)} ms / frame`; });
        if (animate) requestRender();
    }

    // Thumbnails under the Enhancement view: the contribution of every synthesized level (display mode 18),
    // on the same grey scale as the sum. Click: show that level large; click again: back to the sum.
    const thumbs = [0, 1, 2, 3].map((s) => {
        const box = document.createElement('button');
        box.type = 'button';
        box.className = 'pam-thumb';
        box.innerHTML = `<canvas></canvas><span>L${s}</span>`;
        box.addEventListener('click', () => {
            state.enhLevel = state.enhLevel === s ? -1 : s;
            updateControls();
            requestRender();
        });
        $('enhLevels').appendChild(box);
        const cv = box.querySelector('canvas');
        const ctx = cv.getContext('webgpu');
        ctx.configure({ device, format: canvasFormat, alphaMode: 'opaque' });
        return { s, box, cv, ctx, params: uniformBuffer(28 * 4), label: box.querySelector('span') };
    });
    function renderThumbs(encoder, c) {
        const r = consts.ratios, dpr = window.devicePixelRatio || 1;
        for (const t of thumbs) {
            const cssW = t.cv.clientWidth || 200;
            const cw = Math.max(1, Math.round(cssW * dpr)), ch = Math.max(1, Math.round(cssW * img.h / img.w * dpr));
            if (t.cv.width !== cw || t.cv.height !== ch) { t.cv.width = cw; t.cv.height = ch; }
            t.cv.style.height = `${cssW * img.h / img.w}px`;
            device.queue.writeBuffer(t.params, 0, new Float32Array([
                img.w, img.h, SCREEN_WIDTH_PX / SCREEN_WIDTH_CM, consts.sigma0,
                state.gaze[0], state.gaze[1], state.uniformLevel, img.maxLayer,
                0, 0, state.distance, 18,
                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
                cw, ch, 0, 0,
                t.s, 4, 0, c,
            ]));
            const group = bindGroup(displayLayout, [[0, t.params], [1, img.ycc], [2, img.fov], [3, img.maps], [4, img.fovPyr],
                [5, img.bands[Math.min(t.s, img.bands.length - 1)]], [6, img.final], [7, bandStdsBuffer], [8, fitBuffer]]);
            const pass = encoder.beginRenderPass({
                colorAttachments: [{ view: t.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
            });
            pass.setPipeline(displayPipeline);
            pass.setBindGroup(0, group);
            pass.draw(3);
            pass.end();
            t.label.textContent = t.s < c ? `L${t.s}` : `L${t.s} · not synthesized`;
            t.box.classList.toggle('on', state.enhLevel === t.s);
        }
    }

    // Intensity fit plot (SVG): band std per level (point size = weight in the fit), the fitted line, and the noise
    // amplitude of the synthesized levels (levels below the cutoff).
    function drawFitPlot() {
        const g = state.gpuInfo, el = $('fitPlot');
        if (!g || !el) return;
        const n = g.lapStds.length, c = g.cutoff;
        const W = 560, H = 210, L = 46, R = 16, T = 14, B = 34;
        const yMax = Math.max(...g.lapStds, g.m * (n - 0.5) + g.b, 1e-6) * 1.15;
        const X = (lv) => L + (lv + 0.5) / n * (W - L - R);
        const Y = (v) => T + (1 - v / yMax) * (H - T - B);
        const parts = [];
        // levels that get noise
        if (c > 0) parts.push(`<rect x="${X(-0.5)}" y="${T}" width="${X(c - 0.5) - X(-0.5)}" height="${H - T - B}" fill="currentColor" opacity="0.06"/>`,
            `<text x="${(X(-0.5) + X(c - 0.5)) / 2}" y="${T + 12}" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">synthesized levels</text>`);
        // axes
        parts.push(`<path d="M${L} ${T} V${H - B} H${W - R}" fill="none" stroke="currentColor" stroke-width="1" opacity="0.6"/>`);
        for (let i = 0; i < n; i++) parts.push(`<text x="${X(i)}" y="${H - B + 16}" text-anchor="middle" font-size="11" fill="currentColor">L${i}</text>`);
        parts.push(`<text x="${L - 6}" y="${Y(0) + 4}" text-anchor="end" font-size="11" fill="currentColor">0</text>`,
            `<text x="${L - 6}" y="${Y(yMax / 1.15) + 4}" text-anchor="end" font-size="11" fill="currentColor">${(yMax / 1.15).toFixed(3)}</text>`,
            `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">pyramid level · std of the Laplacian band</text>`);
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
        const cssWidth = canvas.parentElement.clientWidth;
        canvas.style.width = `${cssWidth}px`;
        canvas.style.height = `${cssWidth * img.h / img.w}px`;
        canvas.width = Math.round(cssWidth * dpr);
        canvas.height = Math.round(cssWidth * img.h / img.w * dpr);
        requestRender();
    }

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
            chip.addEventListener('click', () => { state.stage = si; state.view = vi; updateControls(); requestRender(); });
            node.querySelector('.chips').appendChild(chip);
        });
        $('flow').appendChild(node);
    });

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

    $('level').addEventListener('input', (e) => { state.level = +e.target.value; updateControls(); requestRender(); });
    document.querySelectorAll('#variantBox button').forEach((b) => b.addEventListener('click', () => {
        state.variant = b.dataset.variant; updateControls(); requestRender();
    }));
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
    $('uniform').addEventListener('change', (e) => { state.uniformLevel = +e.target.value; requestRender(); });
    $('gain').addEventListener('input', (e) => {
        state.gain = +e.target.value;
        $('gainVal').textContent = state.gain.toFixed(2);
        requestRender();
    });
    $('autoGaze').addEventListener('change', (e) => { state.autoGaze = e.target.checked; requestRender(); });
    $('motion').addEventListener('change', (e) => { state.motion = e.target.checked; requestRender(); });
    $('showGaze').addEventListener('change', (e) => { state.showGaze = e.target.checked; requestRender(); });
    $('image').addEventListener('change', (e) => loadImage(e.target.value).catch((err) => status(err.message, true)));
    window.addEventListener('resize', resizeCanvas);

    updateControls();
    await loadImage($('image').value);
}

main().catch((err) => status(`Error: ${err.message}`, true));
