// Calibration study viewer: the study display seen from the observer's eye, with the measured detection rates.
// Geometry and data come from the dataset (data/geometry.json, data/*.csv), previews from previews/<file>.webp.
// Units: cm. The eye sits at the origin and looks along -z; the main display lies in the plane z = -D.
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

const root = document.querySelector('[data-pam-study]');
const BASE = root.dataset.base || '';
const $ = (id) => document.getElementById(id);

// methods: Foveation, Foveation + CE, and ours with an increasing set of recovered statistics (the variants);
// 'family' groups what the chart compares: the two foveation baselines, and the three variants of ours
const METHODS = [
    { key: 'foveated_noCE', family: 'foveation', label: 'No Contrast Enhancement', color: '#bf0c4c' },
    { key: 'foveated_CE', family: 'foveation', label: 'Contrast Enhanced', color: '#6e072c' },
    { key: 'intensity_adjusted', family: 'ours', label: 'Intensity', color: '#deaf6b' },
    { key: 'oriented', family: 'ours', label: '+ Orientation', color: '#de8400' },
    { key: 'phase_aligned', family: 'ours', label: '+ Phase (Ours)', color: '#377e8e' },
];
const familyOf = (key) => METHODS.find((m) => m.key === key).family;
const METHOD_ROW = { foveated_noCE: 'Foveation', foveated_CE: 'Foveation + CE', ours: 'Enhancement' };   // 'ours': the three enhancement variants
let lastOurs = 'phase_aligned';                                    // variant of ours, remembered when switching away
const SCENES = ['Classroom', 'Face', 'Fish', 'Flat', 'House', 'Tree'];
const ECCS = [0, 8, 16, 24];
const LEVELS = [1, 2, 3, 4, 5, 6, 7];
const GREY = 127 / 255;                       // background and mask grey of the study (Unity materials)

const state = { scene: 'Tree', method: 'phase_aligned', ecc: 16, level: 3, refTop: true, rings: true, eyePos: 'centre', autoLevel: false, view2d: false };
// the selection can be linked: #scene=Fish&method=oriented&ecc=24&level=4
for (const [k, v] of new URLSearchParams(location.hash.slice(1))) {
    if (k === 'ecc' || k === 'level') state[k] = +v;
    else if (k === 'scene' || k === 'method') state[k] = v;
}

// ------------------------------------------------------------------ data

function parseCSV(text) {
    const [head, ...rows] = text.trim().split(/\r?\n/);
    const keys = head.split(',');
    return rows.map((r) => {
        const v = r.split(','), o = {};
        keys.forEach((k, i) => { o[k] = isNaN(+v[i]) || v[i] === '' ? v[i] : +v[i]; });
        return o;
    });
}

const DATA_VERSION = 5;   // bump when data/ changes (cache busting)
const [geo, perImage, perCond, thresholds] = await Promise.all([
    fetch(BASE + 'data/geometry.json?v=' + DATA_VERSION).then((r) => r.json()),
    fetch(BASE + 'data/per_image.csv?v=' + DATA_VERSION).then((r) => r.text()).then(parseCSV),
    fetch(BASE + 'data/per_condition.csv?v=' + DATA_VERSION).then((r) => r.text()).then(parseCSV),
    fetch(BASE + 'data/thresholds.csv?v=' + DATA_VERSION).then((r) => r.text()).then(parseCSV),
]);

const findImage = (s) => perImage.find((r) => r.scene === s.scene && r.method === s.method && r.level === s.level && r.mask_ecc_deg === s.ecc);
const findCond = (m, l, e) => perCond.find((r) => r.method === m && r.level === l && r.mask_ecc_deg === e);
const findThr = (m, e) => thresholds.find((r) => r.method === m && r.mask_ecc_deg === e);
const preview = (file) => BASE + 'previews/' + file.replace(/\.png$/, '.webp');

// ------------------------------------------------------------------ geometry

const D = geo.observer.viewing_distance_cm;
const PITCH = geo.display.pixel_pitch_mm / 10;              // cm per screen pixel
const [W_PX, H_PX] = geo.display.resolution_px;

function viewing(ecc) {
    if (ecc === 0) return { eyePx: W_PX / 2, fixPx: W_PX / 2, mask: null };
    return { eyePx: geo.observer.eye_position_px[0], fixPx: geo.conditions[ecc].fixation_px[0], mask: geo.foveal_mask.conditions[ecc] };
}
const pxToWorld = (px, py, eyePx) => new THREE.Vector3((px - eyePx) * PITCH, (H_PX / 2 - py) * PITCH, -D);

// ------------------------------------------------------------------ three.js scene

THREE.ColorManagement.enabled = false;                         // images and colours are used as they are (sRGB in, sRGB out)
const canvas = $('psCanvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene3 = new THREE.Scene();
scene3.background = new THREE.Color('#e9e6dc');
const camera = new THREE.PerspectiveCamera(46, 3, 1, 5000);
camera.rotation.order = 'YXZ';

// room: floor grid, so the view reads as a space around the observer
const floor = new THREE.GridHelper(1200, 48, 0xc9c2b2, 0xd8d2c4);
floor.position.y = -115;
scene3.add(floor);

// main display: bezel + screen (shader draws the two halves, the margins and the foveal mask, as Unity did)
const screenUniforms = {
    tTop: { value: null }, tBot: { value: null }, mirror: { value: 1 },
    maskOn: { value: 0 }, maskC: { value: new THREE.Vector2() }, rIn: { value: 1 }, rOut: { value: 2 },
    fixOn: { value: 0 }, fixP: { value: new THREE.Vector2() }, grey: { value: GREY },
};
const screen = new THREE.Mesh(new THREE.PlaneGeometry(W_PX * PITCH, H_PX * PITCH), new THREE.ShaderMaterial({
    uniforms: screenUniforms,
    vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `
        uniform sampler2D tTop, tBot; uniform float mirror, maskOn, rIn, rOut, fixOn, grey; uniform vec2 maskC, fixP;
        varying vec2 vUv;
        void main() {
            vec2 p = vec2(vUv.x * ${W_PX}.0, (1.0 - vUv.y) * ${H_PX}.0);      // screen pixel, y down
            vec3 col = vec3(grey);
            float u = p.x / ${W_PX}.0;
            if (mirror > 0.5) u = 1.0 - u;                                     // left-side sessions: image mirrored
            if (p.y >= 56.0 && p.y < 1080.0) col = texture2D(tTop, vec2(u, (p.y - 56.0) / 1024.0)).rgb;
            else if (p.y >= 1080.0 && p.y < 2104.0) col = texture2D(tBot, vec2(u, (2104.0 - p.y) / 1024.0)).rgb;  // mirrored at the midline
            // grey bar over the boundary between the halves (Unity 'Bar (1)'); smoothstep fitted to the sprite's alpha profile
            col = mix(col, vec3(grey), 1.0 - smoothstep(40.5, 71.5, abs(p.y - 1080.0)));
            if (maskOn > 0.5) col = mix(col, vec3(grey), clamp((rOut - distance(p, maskC)) / (rOut - rIn), 0.0, 1.0));
            if (fixOn > 0.5 && distance(p, fixP) < 9.0) col = vec3(0.749, 0.047, 0.298);   // #bf0c4c, as the dot and rings
            gl_FragColor = vec4(col, 1.0);
        }`,
}));
const bezel = new THREE.Mesh(new THREE.BoxGeometry(W_PX * PITCH + 2.4, H_PX * PITCH + 2.4, 3), new THREE.MeshBasicMaterial({ color: 0x18181a }));
const stand = new THREE.Mesh(new THREE.BoxGeometry(8, 115 - H_PX * PITCH / 2, 4), new THREE.MeshBasicMaterial({ color: 0x2a2a2e }));
scene3.add(screen, bezel, stand);

// fixation point: a dot with a label (on the rim of the display for 8°, beside it for 16° and 24°).
// The dot keeps the same size on screen at any zoom: its radius is 1 and it is scaled every frame (setFixDotSize).
const FIX_DOT_PX = 4;                                               // radius in CSS px
const fixation = new THREE.Group();
const fixDot = new THREE.Mesh(new THREE.CircleGeometry(1, 32), new THREE.MeshBasicMaterial({ color: 0xbf0c4c, depthTest: false }));
fixDot.renderOrder = 12;
fixation.add(fixDot);
scene3.add(fixation);

// Labels are HTML text over the view (sharp at any size, in the page font), placed every frame at the screen position of
// a 3D point. The ring lines themselves are WebGL lines.
const labelLayer = document.createElement('div');
labelLayer.className = 'ps-labels';
canvas.parentElement.insertBefore(labelLayer, canvas.nextSibling);
const labels = [];                                                  // { el, obj: parent Object3D, pos: local position, group }
function addLabel(text, cls, obj, pos, group) {
    const el = document.createElement('span');
    el.className = cls;
    el.textContent = text;
    labelLayer.appendChild(el);
    labels.push({ el, obj, pos, group });
}
const _p = new THREE.Vector3();
function placeLabels() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    for (const l of labels) {
        let shown = true;
        for (let o = l.obj; o; o = o.parent) if (!o.visible) shown = false;
        _p.copy(l.pos).applyMatrix4(l.obj.matrixWorld).project(camera);
        if (!shown || _p.z < -1 || _p.z > 1 || Math.abs(_p.x) > 1.2 || Math.abs(_p.y) > 1.2) { l.el.hidden = true; continue; }
        l.el.hidden = false;
        l.el.style.transform = `translate(${((_p.x + 1) / 2) * w}px, ${((1 - _p.y) / 2) * h}px)`;
    }
}
function removeLabels(group) {
    for (let i = labels.length - 1; i >= 0; i--) if (labels[i].group === group) { labels[i].el.remove(); labels.splice(i, 1); }
}

// eccentricity rings around the line of sight
const rings = new THREE.Group();
scene3.add(rings);
function buildRings(gaze) {
    rings.clear();
    removeLabels('rings');
    const R = 400, up = new THREE.Vector3(0, 1, 0);
    const u = new THREE.Vector3().crossVectors(gaze, up).normalize(), v = new THREE.Vector3().crossVectors(u, gaze).normalize();
    const mat = new THREE.LineBasicMaterial({ color: 0xbf0c4c, transparent: true, opacity: 0.55, depthTest: false });
    for (let e = 10; e <= 90; e += 10) {
        const a = THREE.MathUtils.degToRad(e), pts = [];
        for (let i = 0; i <= 128; i++) {
            const phi = (i / 128) * Math.PI * 2;
            pts.push(gaze.clone().multiplyScalar(Math.cos(a))
                .addScaledVector(u, Math.sin(a) * Math.cos(phi)).addScaledVector(v, Math.sin(a) * Math.sin(phi)).multiplyScalar(R));
        }
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat);
        line.renderOrder = 10;
        rings.add(line);
        const phi = Math.PI * 0.12;                                // label a little above the horizontal, on the far side
        const pos = gaze.clone().multiplyScalar(Math.cos(a))
            .addScaledVector(u, Math.sin(a) * Math.cos(phi)).addScaledVector(v, Math.sin(a) * Math.sin(phi)).multiplyScalar(R);
        addLabel(`${e}°`, 'ps-lbl-ring', rings, pos, 'rings');
    }
}

addLabel('fixation point', 'ps-lbl-fix', fixation, new THREE.Vector3(0, 0, 0), 'fix');

// ------------------------------------------------------------------ view

const HFOV = 104;                                                  // horizontal field of view at the start
const view = { yaw: 0, pitch: 0, hfov: HFOV };
// True Size: the field of view that shows the scene at its real angular size for the person in front of the screen.
// The physical width of the view follows from the measured width of the whole screen (CSS px of the view / of the
// screen), so it stays right in a window and in full screen. Kept in this browser.
const setup = { screenCm: window.screen.width * 2.54 / 96, distance: 60, trueSize: false };
try {
    const s = JSON.parse(localStorage.getItem('ps-screen') || '{}');
    if (s.screenCm > 0) setup.screenCm = s.screenCm;
    if (s.distance > 0) setup.distance = s.distance;
} catch (e) { /* storage unavailable: defaults */ }
const saveSetup = () => { try { localStorage.setItem('ps-screen', JSON.stringify({ screenCm: setup.screenCm, distance: setup.distance })); } catch (e) { /* ignore */ } };
const deg = THREE.MathUtils.radToDeg, rad = THREE.MathUtils.degToRad;
const imageWidthCm = () => setup.screenCm * canvas.clientWidth / window.screen.width;
const trueHfov = () => deg(2 * Math.atan(imageWidthCm() / 2 / setup.distance));
const vfov = () => deg(2 * Math.atan(Math.tan(rad(view.hfov) / 2) * canvas.clientHeight / canvas.clientWidth));
let gazeDir = new THREE.Vector3(0, 0, -1);

function lookAlong(dir) {
    view.yaw = Math.atan2(-dir.x, -dir.z);
    view.pitch = Math.asin(THREE.MathUtils.clamp(dir.y, -1, 1));
}

// Right Rim shows the right-side sessions: the whole setup mirrored (the images are then not mirrored, as in Unity)
const mirrorSign = () => (state.eyePos === 'right' ? -1 : 1);

function layout() {
    scene3.scale.x = mirrorSign();
    labelLayer.dataset.eye = state.eyePos;                         // CSS puts the fixation label beside the dot on a rim
    const { eyePx, fixPx, mask } = viewing(state.ecc);
    const centre = pxToWorld(W_PX / 2, H_PX / 2, eyePx);
    screen.position.copy(centre);
    bezel.position.copy(centre).add(new THREE.Vector3(0, 0, -1.6));
    stand.position.set(centre.x, -115 + (115 - H_PX * PITCH / 2) / 2, -D - 6);

    screenUniforms.mirror.value = state.ecc === 0 ? 0 : 1;
    screenUniforms.maskOn.value = mask ? 1 : 0;
    if (mask) {
        screenUniforms.maskC.value.set(...mask.centre_px);
        screenUniforms.rIn.value = mask.radius_px.opaque_until;
        screenUniforms.rOut.value = mask.radius_px.fully_transparent_from;
    }
    screenUniforms.fixOn.value = 0;

    const fixWorld = pxToWorld(fixPx, H_PX / 2, eyePx);
    fixation.visible = state.ecc !== 0;
    fixation.position.copy(fixWorld).add(new THREE.Vector3(0, 0, 0.5));
    fixation.lookAt(0, 0, 0);                                     // the dot faces the observer

    gazeDir = (state.ecc === 0 ? new THREE.Vector3(0, 0, -1) : fixWorld.clone()).normalize();
    buildRings(gazeDir);
    rings.visible = state.rings;
    lookAlong(gazeDir.clone().multiply(new THREE.Vector3(mirrorSign(), 1, 1)));   // start looking at the fixation point
}

function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;          // 3:1 on the page (CSS), the whole screen in full screen
    renderer.setSize(w, h, false);
    if (setup.trueSize) view.hfov = trueHfov();
    syncSetup();
    render();
}

const setFixDotSize = (worldPerPx) => fixDot.scale.setScalar(FIX_DOT_PX * worldPerPx);

let pending = false;
function render() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
        pending = false;
        // Centre: the eye in front of the centre of the image. Left / Right Rim: off-axis projection for an eye straight in
        // front of that edge of the image (the image is one half of a frustum twice as wide, centred on the rim); the
        // fixation point appears on the rim and the view stays perpendicular to the screen.
        const w = canvas.clientWidth, h = canvas.clientHeight;
        if (state.eyePos === 'centre') {
            camera.clearViewOffset();
            camera.aspect = w / h;
        } else {
            camera.aspect = (2 * w) / h;
            camera.setViewOffset(2 * w, h, state.eyePos === 'left' ? w : 0, 0, w, h);
        }
        camera.fov = vfov();                                       // focal length from the field of view (or True Size)
        camera.updateProjectionMatrix();
        camera.rotation.set(view.pitch, view.yaw, 0);
        // world size of one CSS px at the dot's distance from the eye (the camera sits at the eye)
        setFixDotSize(fixation.position.length() * 2 * Math.tan(rad(camera.fov) / 2) / h);
        if (state.view2d) {
            render2d(w, h);
            return;
        }
        labelLayer.hidden = false;
        renderer.render(scene3, camera);
        placeLabels();
    });
}

// 2D view: the image on the display (both halves, 3840 x 2048 px), front-on with an orthographic camera, filling the
// view (the canvas switches to 3840:2048, see CSS); no bezel, stand, rings, labels or room
const cam2d = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 1000);
const _c = new THREE.Vector3();
function render2d(w, h) {
    scene3.updateMatrixWorld();
    screen.getWorldPosition(_c);
    cam2d.position.set(_c.x, _c.y, _c.z + 200);
    cam2d.lookAt(_c);
    const imgW = W_PX * PITCH, imgH = 2048 * PITCH, a = w / h;        // both halves: 3840 x 2048 px
    const halfH = a > imgW / imgH ? imgH / 2 : imgW / 2 / a;
    cam2d.left = -halfH * a; cam2d.right = halfH * a; cam2d.top = halfH; cam2d.bottom = -halfH;
    cam2d.updateProjectionMatrix();
    setFixDotSize(2 * halfH / h);                                    // world size of one CSS px in the 2D view
    const shown = [rings, bezel, stand, floor].map((o) => o.visible);
    for (const o of [rings, bezel, stand, floor]) o.visible = false;
    const bg = scene3.background;
    scene3.background = new THREE.Color(GREY, GREY, GREY);
    labelLayer.hidden = true;
    renderer.render(scene3, cam2d);
    [rings, bezel, stand, floor].forEach((o, i) => { o.visible = shown[i]; });
    scene3.background = bg;
}

// drag to look around, wheel or pinch to zoom (3D view only)
let drag = null;
canvas.addEventListener('pointerdown', (e) => { if (state.view2d) return; drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); canvas.classList.add('ps-grabbing'); });
canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const k = rad(vfov()) / canvas.clientHeight;
    view.yaw += (e.clientX - drag.x) * k;
    view.pitch = THREE.MathUtils.clamp(view.pitch + (e.clientY - drag.y) * k, -1.4, 1.4);
    drag = { x: e.clientX, y: e.clientY };
    render();
});
const endDrag = () => { drag = null; canvas.classList.remove('ps-grabbing'); };
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);
canvas.addEventListener('wheel', (e) => {
    if (state.view2d) return;
    e.preventDefault();
    setFov(view.hfov * Math.exp(e.deltaY * 0.001));
}, { passive: false });

// ------------------------------------------------------------------ textures

const loader = new THREE.TextureLoader();
const cache = new Map();
function texture(file) {
    if (!cache.has(file)) {
        cache.set(file, new Promise((resolve) => loader.load(preview(file), (t) => {
            t.flipY = false;
            t.colorSpace = THREE.NoColorSpace;
            t.anisotropy = renderer.capabilities.getMaxAnisotropy();
            resolve(t);
        }, undefined, () => resolve(null))));
    }
    return cache.get(file);
}

let loadToken = 0;
async function updateTextures() {
    const row = findImage(state);
    const token = ++loadToken;
    const refFile = `${state.scene.toLowerCase()}_reference.png`;
    const [ref, test] = await Promise.all([texture(refFile), row ? texture(row.image_file) : Promise.resolve(null)]);
    if (token !== loadToken) return;
    screenUniforms.tTop.value = state.refTop ? ref : (test || ref);
    screenUniforms.tBot.value = state.refTop ? (test || ref) : ref;
    render();
}

// ------------------------------------------------------------------ controls

function segment(boxId, items, pick, label) {
    const box = $(boxId);
    box.innerHTML = '';
    for (const v of items) {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = label(v);
        b.dataset.value = v;
        b.addEventListener('click', () => pick(v));
        box.appendChild(b);
    }
}
segment('psScene', SCENES, (v) => { state.scene = v; update('scene'); }, (s) => s);
segment('psFamily', Object.keys(METHOD_ROW), (v) => { state.method = v === 'ours' ? lastOurs : v; update('method'); }, (v) => METHOD_ROW[v]);
segment('psMethod', METHODS.filter((m) => m.family === 'ours').map((m) => m.key), (v) => { state.method = v; update('method'); },
    (k) => METHODS.find((m) => m.key === k).label);
segment('psEcc', ECCS, (v) => { state.ecc = v; update('ecc'); }, (e) => (e === 0 ? 'Free Viewing' : `${e}°`));
segment('psLevel', LEVELS, (v) => { state.level = v; state.autoLevel = false; update('level'); }, (l) => `L${l}`);
segment('psRef', ['top', 'bottom'], (v) => { state.refTop = v === 'top'; update('refTop'); }, (v) => (v === 'top' ? 'Top' : 'Bottom'));

$('psAutoLevel').addEventListener('click', () => { state.autoLevel = !state.autoLevel; update('level'); });
$('psView2d').addEventListener('click', () => { state.view2d = !state.view2d; syncButtons(); resize(); });
$('psResetView').addEventListener('click', () => { if (!setup.trueSize) view.hfov = HFOV; layout(); syncSetup(); render(); });

// scrolling zooms (and leaves True Size); the True Size panel sets the field of view from screen width and distance
function setFov(h) {
    view.hfov = THREE.MathUtils.clamp(h, 10, 150);
    setup.trueSize = false;
    syncSetup();
    render();
}
function syncSetup() {
    for (const [id, v] of [['psScreenCm', setup.screenCm.toFixed(1)], ['psDist', String(setup.distance)]]) {
        if (document.activeElement !== $(id)) $(id).value = v;          // do not overwrite while typing
    }
    $('psTrue').classList.toggle('on', setup.trueSize);
    $('psEyePos').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.value === state.eyePos));
    $('psSetupBtn').classList.toggle('ps-active', setup.trueSize);
    const where = { centre: 'the centre', left: 'the left edge', right: 'the right edge' }[state.eyePos];
    $('psFovInfo').textContent = setup.trueSize
        ? `Sit ${setup.distance} cm straight in front of ${where} of the image (${imageWidthCm().toFixed(1)} cm wide).`
        : `Zoom: ${view.hfov.toFixed(1)}° for the whole width (centre view).`;
}
const onSetupInput = () => {
    const cm = +$('psScreenCm').value, d = +$('psDist').value;
    if (cm > 0) setup.screenCm = cm;
    if (d > 0) setup.distance = d;
    saveSetup();
    if (setup.trueSize) view.hfov = trueHfov();
    syncSetup();
    render();
};
$('psScreenCm').addEventListener('input', onSetupInput);
$('psDist').addEventListener('input', onSetupInput);
$('psSetupBtn').addEventListener('click', () => {
    const open = $('psSetup').hidden;
    $('psSetup').hidden = !open;
    $('psSetupBtn').classList.toggle('on', open);
    $('psSetupBtn').setAttribute('aria-expanded', String(open));
    if (open) syncSetup();
});
segment('psEyePos', ['centre', 'left', 'right'], (v) => { state.eyePos = v; layout(); syncSetup(); render(); },
    (v) => ({ centre: 'Centre', left: 'Left Rim', right: 'Right Rim' })[v]);
$('psTrue').addEventListener('click', () => {
    setup.trueSize = !setup.trueSize;
    view.hfov = setup.trueSize ? trueHfov() : HFOV;
    syncSetup();
    render();
});

// full screen, as in the real-time demo
const FS_ICON = {
    enter: '<svg viewBox="0 0 32 18" width="27.2" height="15.3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 6V1h6M25 1h6v5M31 12v5h-6M7 17H1v-5"/></svg>',
    exit: '<svg viewBox="0 0 32 18" width="27.2" height="15.3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 1v5H1M31 6h-6V1M25 17v-5h6M1 12h6v5"/></svg>',
};
const fsBox = canvas.parentElement;
$('psFullscreen').innerHTML = FS_ICON.enter;
$('psFullscreen').addEventListener('click', () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else if (fsBox.requestFullscreen) fsBox.requestFullscreen().catch(() => {});
});
// in full screen the selection buttons and the chart move into the view (one panel, bottom left); back afterwards
const fsMoved = [root.querySelector('.ps-controls'), $('psChart')].map((el) => {
    const home = document.createComment('ps-home');
    el.before(home);
    return { el, home };
});
const fsPanel = document.createElement('div');
fsPanel.className = 'ps-fs-panel';
document.addEventListener('fullscreenchange', () => {
    const on = document.fullscreenElement === fsBox;
    for (const { el, home } of fsMoved) on ? fsPanel.appendChild(el) : home.after(el);
    if (on) fsBox.appendChild(fsPanel);
    else fsPanel.remove();
    fitFsChart();
    $('psFullscreen').innerHTML = on ? FS_ICON.exit : FS_ICON.enter;
    $('psFullscreen').setAttribute('aria-label', on ? 'Leave full screen' : 'Full screen');
    resize();
});
$('psRingsBtn').addEventListener('click', () => { state.rings = !state.rings; rings.visible = state.rings; syncButtons(); render(); });

function syncButtons() {
    const mark = (id, val) => $(id).querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.value === String(val)));
    mark('psScene', state.scene);
    const ours = familyOf(state.method) === 'ours';
    if (ours) lastOurs = state.method;
    mark('psFamily', ours ? 'ours' : state.method);
    mark('psMethod', ours ? state.method : '');
    // the variants stay clickable: picking one while a foveation method is selected switches to ours with that variant
    mark('psEcc', state.ecc);
    mark('psLevel', state.level);
    mark('psRef', state.refTop ? 'top' : 'bottom');
    const thrLevel = Math.log2(40 / findThr(state.method, state.ecc).threshold_cpd);   // threshold as a (fractional) level
    $('psLevel').querySelectorAll('button').forEach((b) => {
        const l = +b.dataset.value, ok = !!findImage({ ...state, level: l });
        b.disabled = !ok;
        b.classList.toggle('ps-visible', l > thrLevel);           // untested levels too, so the row does not shift
        b.classList.toggle('ps-invisible', l < thrLevel);
        const cpd = `level ${l}: input up to ${+(40 / 2 ** l).toFixed(2)} cpd; `;
        b.title = cpd + (!ok ? 'not tested for this scene, method and mask'
            : l > thrLevel ? 'above the 75 % threshold: observers could tell it from the reference'
            : 'below the 75 % threshold: observers could not reliably tell it from the reference');
    });
    $('psAutoLevel').classList.toggle('on', state.autoLevel);
    $('psView2d').classList.toggle('on', state.view2d);
    for (const id of ['psRingsBtn', 'psSetupBtn', 'psResetView']) $(id).disabled = state.view2d;   // 3D only
    if (state.view2d) $('psSetup').hidden = true;
    canvas.classList.toggle('ps-2d', state.view2d);
    root.querySelector('.ps-hint').hidden = state.view2d;
    $('psRingsBtn').classList.toggle('on', state.rings);
}

// highest tested level below the threshold of this method and mask (or the lowest tested level, if none is below)
function highestInvisible() {
    const thrLevel = Math.log2(40 / findThr(state.method, state.ecc).threshold_cpd);
    const tested = LEVELS.filter((l) => findImage({ ...state, level: l }));
    const below = tested.filter((l) => l < thrLevel);
    return below.length ? Math.max(...below) : Math.min(...tested);
}

function update(changed) {
    if (state.autoLevel) state.level = highestInvisible();
    if (!findImage(state)) {                                        // keep a tested level: the nearest one
        const ok = LEVELS.filter((l) => findImage({ ...state, level: l }));
        if (ok.length) state.level = ok.reduce((a, b) => (Math.abs(b - state.level) < Math.abs(a - state.level) ? b : a));
    }
    if (changed === 'ecc' || changed === 'init') layout();
    if (changed !== 'init') history.replaceState(null, '', `#scene=${state.scene}&method=${state.method}&ecc=${state.ecc}&level=${state.level}`);
    syncButtons();
    updateTextures();
    updateInfo();
    render();
}

// ------------------------------------------------------------------ info and chart

const pct = (p) => `${Math.round(p * 100)} %`;

function updateInfo() {
    const row = findImage(state), m = METHODS.find((x) => x.key === state.method);
    const cond = findCond(state.method, state.level, state.ecc), thr = findThr(state.method, state.ecc);
    let html = '';
    if (row) {
        html += `<p class="ps-big"><b>${pct(row.p_correct)}</b> correct <span>(${row.n_correct} of ${row.n_trials} observers found the reference)</span></p>
            <p>95 % interval over observers: ${pct(row.ci95_low_boot)}–${pct(row.ci95_high_boot)}. Chance is 50 %.</p>`;
    }
    if (cond) html += `<p>All six scenes at this level: ${pct(cond.p_correct)} (${cond.n_correct} of ${cond.n_trials}).</p>`;
    $('psText').innerHTML = html;
    drawChart(m, thr);
}

// Size of the chart in px: beside the buttons as high as they are and as wide as its column; below them the full width;
// in the full-screen panel as high as the buttons. Drawn at that size (no scaling), so text and lines keep their size.
function chartSize() {
    const chart = $('psChart'), controls = root.querySelector('.ps-controls');
    const ch = controls.offsetHeight;
    if (chart.parentElement.classList.contains('ps-fs-panel')) return [Math.round(ch * 1.8), ch];
    const cr = chart.getBoundingClientRect(), kr = controls.getBoundingClientRect();
    const beside = cr.left > kr.right - 1;
    const w = Math.max(200, Math.floor(chart.clientWidth));
    return beside ? [w, ch] : [w, Math.round(Math.min(280, Math.max(180, w * 0.4)))];
}

let chartArgs = null;
function drawChart(m, thr) {
    chartArgs = [m, thr];
    const [W, H] = chartSize();
    const L = 40, R = 1, T = 6, B = 34;                           // plot area flush with the right edge
    const x = (lvl) => L + ((lvl - 0.5) / 7) * (W - L - R);
    const y = (p) => T + (1 - (p - 0.4) / 0.6) * (H - T - B);
    const el = [];
    el.push(`<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="var(--gray-light)" rx="4"/>`);
    // stripes per level interval: where the curve crosses 75 % decides the highest invisible level (the level left of
    // the crossing); the interval with this method's crossing is tinted
    const tl = Math.log2(40 / thr.threshold_cpd);
    for (let n = 0; n <= 7; n++) {
        const x0 = x(Math.max(0.5, n)), x1 = x(Math.min(7.5, n + 1));
        const hit = tl >= n && tl < n + 1;
        if (hit) el.push(`<rect x="${x0}" y="${T}" width="${x1 - x0}" height="${H - T - B}" fill="${m.color}" opacity="0.13"/>`);
        else if (n % 2 === 1) el.push(`<rect x="${x0}" y="${T}" width="${x1 - x0}" height="${H - T - B}" fill="#000" opacity="0.035"/>`);
    }
    for (const p of [0.5, 0.75, 1]) {
        el.push(`<line x1="${L}" x2="${W - R}" y1="${y(p)}" y2="${y(p)}" stroke="#cfc8b8" stroke-dasharray="${p === 0.75 ? '3 3' : ''}"/>`);
        el.push(`<text x="${L - 6}" y="${y(p) + 4}" text-anchor="end">${Math.round(p * 100)}%</text>`);
    }
    for (const l of LEVELS) el.push(`<text x="${x(l)}" y="${H - B + 15}" text-anchor="middle">L${l}</text>`);
    el.push(`<text x="${(L + W - R) / 2}" y="${H - 3}" text-anchor="middle">level (input resolution halved per level)</text>`);
    // fitted psychometric functions: the other variants of the same group faint, the selected one on top
    const curve = (t) => {
        const pts = [];
        for (let l = 0.5; l <= 7.5; l += 0.05) {
            const c = 40 / 2 ** l, p = 1 - 0.5 * Math.exp(-(((Math.log2(40) - Math.log2(c)) / t.weibull_l) ** t.weibull_k));
            pts.push(`${x(l).toFixed(1)},${y(Math.max(0.4, p)).toFixed(1)}`);
        }
        return pts.join(' ');
    };
    for (const o of METHODS) {
        if (o.key !== m.key && o.family === m.family) el.push(`<polyline points="${curve(findThr(o.key, state.ecc))}" fill="none" stroke="${o.color}" stroke-width="1.5" opacity="0.3"/>`);
    }
    el.push(`<polyline points="${curve(thr)}" fill="none" stroke="${m.color}" stroke-width="2"/>`);
    el.push(`<line x1="${x(tl)}" x2="${x(tl)}" y1="${T}" y2="${H - B}" stroke="${m.color}" stroke-width="1.5" stroke-dasharray="4 3"/>`);
    // all scenes per level (with interval), this scene as small rings
    for (const l of LEVELS) {
        const c = findCond(state.method, l, state.ecc);
        if (c) {
            el.push(`<line x1="${x(l)}" x2="${x(l)}" y1="${y(Math.max(0.4, c.ci95_low_boot))}" y2="${y(c.ci95_high_boot)}" stroke="${m.color}" stroke-width="1.5"/>`);
            el.push(`<circle cx="${x(l)}" cy="${y(Math.max(0.4, c.p_correct))}" r="4.5" fill="${m.color}"/>`);
        }
        const r = findImage({ ...state, level: l });
        if (r) el.push(`<circle cx="${x(l) + 9}" cy="${y(Math.max(0.4, r.p_correct))}" r="3" fill="white" stroke="#4f4f55" stroke-width="1.3"/>`);
    }
    // legend in the bottom-right corner of the plot area, below the 50 % line (the plot reaches down to 40 %)
    const lx = W - R - 8, ly = H - B - 4;
    el.push(`<g class="ps-plotlegend" text-anchor="end">
        <text x="${lx - 10}" y="${ly - 10}">all scenes</text><circle cx="${lx - 3}" cy="${ly - 13}" r="3" fill="${m.color}"/>
        <text x="${lx - 10}" y="${ly}">${state.scene}</text><circle cx="${lx - 3}" cy="${ly - 3}" r="2.6" fill="white" stroke="#4f4f55" stroke-width="1.1"/>
    </g>`);
    el.push(`<circle cx="${x(state.level)}" cy="${y(Math.max(0.4, (findCond(state.method, state.level, state.ecc) || { p_correct: 0.4 }).p_correct))}" r="8.5" fill="none" stroke="#4f4f55" stroke-width="1.5"/>`);
    $('psChart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:${W}px;height:${H}px" role="img" aria-label="proportion correct per level">${el.join('')}</svg>`;
}
// redraw at the new size when the layout changes (window, full screen, buttons wrapping)
let chartPending = false;
function fitFsChart() {
    if (chartPending || !chartArgs) return;
    chartPending = true;
    requestAnimationFrame(() => { chartPending = false; drawChart(...chartArgs); });
}
new ResizeObserver(fitFsChart).observe(root.querySelector('.ps-main'));

// ------------------------------------------------------------------ start

new ResizeObserver(resize).observe(canvas);
update('init');
resize();
root.classList.add('ps-ready');
