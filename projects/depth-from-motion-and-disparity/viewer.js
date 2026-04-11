import * as THREE from 'https://esm.sh/three@0.160.0?bundle';
import { OrbitControls } from 'https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js?bundle';
import { OBJLoader } from 'https://esm.sh/three@0.160.0/examples/jsm/loaders/OBJLoader.js?bundle';

const canvas = document.getElementById('c');
const viewer = document.getElementById('viewer');
const viewerOverlay = document.getElementById('viewer-overlay');

function isOverlayVisible() {
    return Boolean(viewerOverlay) && !viewerOverlay.classList.contains('is-hidden');
}

const INACTIVITY_TIMEOUT_MS = 5000;
let inactivityTimerId = null;
let isPointerDown = false;
let isDraggingWithControls = false;

function isActivelyInteracting() {
    return isPointerDown || isDraggingWithControls;
}

function clearOverlayTimer() {
    if (inactivityTimerId !== null) {
        window.clearTimeout(inactivityTimerId);
        inactivityTimerId = null;
    }
}

function showViewerOverlay() {
    if (!viewerOverlay) {
        return;
    }

    viewerOverlay.classList.remove('is-hidden');
}

function dismissViewerOverlay() {
    if (!viewerOverlay || viewerOverlay.classList.contains('is-hidden')) {
        return;
    }

    viewerOverlay.classList.add('is-hidden');
}

function scheduleOverlayReturn() {
    if (!viewerOverlay) {
        return;
    }

    clearOverlayTimer();

    inactivityTimerId = window.setTimeout(() => {
        if (!isActivelyInteracting()) {
            showViewerOverlay();
        }
    }, INACTIVITY_TIMEOUT_MS);
}

function handleTransientInteraction() {
    dismissViewerOverlay();
    scheduleOverlayReturn();
}

function beginContinuousInteraction() {
    clearOverlayTimer();
}

function endContinuousInteraction() {
    if (!isActivelyInteracting()) {
        scheduleOverlayReturn();
    }
}

function getRenderSize() {
    // Use actual on-screen canvas size so max-width and responsive CSS stay in sync.
    const width = canvas.clientWidth || viewer.clientWidth;
    const height = canvas.clientHeight || viewer.clientHeight;
    return { width, height };
}

const initialSize = getRenderSize();

// Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

// Camera
const aspect = initialSize.width / initialSize.height;
const frustumSize = 5;
const camera = new THREE.OrthographicCamera((frustumSize * aspect) / -2,
                                             (frustumSize * aspect) / 2,
                                             frustumSize / 2,
                                             frustumSize / -2,
                                             0.1,
                                             1000);
camera.position.z = 3;

// Renderer
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

function resizeRenderer() {
    const { width, height } = getRenderSize();
    const aspect = width / height;

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height, false);

    camera.left = (-frustumSize * width / height) / 2;
    camera.right = (frustumSize * width / height) / 2;
    camera.top = frustumSize / 2;
    camera.bottom = -frustumSize / 2;

    camera.updateProjectionMatrix();
}
resizeRenderer();

const controls = new OrbitControls(camera, renderer.domElement);
controls.minPolarAngle = Math.PI / 2 - Math.PI / 12; // 45° up
controls.maxPolarAngle = Math.PI / 2 + Math.PI / 12; // 45° down
controls.minAzimuthAngle = -Math.PI / 12; // left 45°
controls.maxAzimuthAngle = Math.PI / 12; // right 45°
controls.enableZoom = false;
controls.rotateSpeed = 0.35;
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = isOverlayVisible();
const overlayAutoRotateSpeed = 1.8;
controls.autoRotateSpeed = overlayAutoRotateSpeed;



controls.addEventListener('start', () => {
    isDraggingWithControls = true;
    beginContinuousInteraction();
});
controls.addEventListener('end', () => {
    isDraggingWithControls = false;
    endContinuousInteraction();
});
canvas.addEventListener('pointerdown', () => {
    dismissViewerOverlay();
    isPointerDown = true;
    beginContinuousInteraction();
}, { passive: true });
window.addEventListener('pointerup', () => {
    isPointerDown = false;
    endContinuousInteraction();
});
window.addEventListener('pointercancel', () => {
    isPointerDown = false;
    endContinuousInteraction();
});


// Cylinder
const geometry_cylinder = new THREE.CylinderGeometry(
    0.05, // top radius
    0.05, // bottom radius
    2, // height
    32 // radial segments (smoothness)
);
const material_cylinder = new THREE.MeshStandardMaterial({
    color: 0x111111, // near-black (pure black can look flat/undefined)
    roughness: 0.3, // MAX matte
    metalness: 0.0 // no metallic shine
});
const cylinder = new THREE.Mesh(geometry_cylinder, material_cylinder);
scene.add(cylinder);
cylinder.rotation.set(Math.PI / 2, 0, 0); // Rotate the cylinder to lie flat on the XZ plane

const geometry_sphere = new THREE.SphereGeometry(0.14, 32, 32);
const sphere = new THREE.Mesh(geometry_sphere, material_cylinder);
// sphere.position.set(0, 0, 0);
sphere.scale.set(2,2,0.3);
scene.add(sphere);



// Material for Object
const textureLoader = new THREE.TextureLoader();
const texture = textureLoader.load('/projects/depth-from-motion-and-disparity/texture.png');
texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
texture.minFilter = THREE.LinearMipmapLinearFilter;
texture.magFilter = THREE.LinearFilter;
texture.generateMipmaps = true;
texture.needsUpdate = true;

const material = new THREE.MeshBasicMaterial({
    map: texture
});


// Load own object
const loader = new OBJLoader();
loader.load(
    '/projects/depth-from-motion-and-disparity/stimulus_loRes.obj',

    (object) => {
        scene.add(object);

        // optional: scale it (VERY common fix)
        object.scale.set(0.5, 0.5, 0.5);

        // optional: center it
        const box = new THREE.Box3().setFromObject(object);
        const center = box.getCenter(new THREE.Vector3());
        object.position.sub(center);

        object.rotation.set(Math.PI / 2, 0, 0); // Rotate the object to lie flat on the XZ plane
        object.scale.set(1, 1, 1);

        object.traverse((child) => {
            if (child.isMesh) {
                child.material = material;
            }
        });
    },

    (xhr) => {
        console.log((xhr.loaded / xhr.total) * 100 + '% loaded');
    },

    (error) => {
        console.error('OBJ load error:', error);
    }
);

// Light
const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(0, 0, 5);
scene.add(light);

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    // cube.rotation.x += 0.01;
    // cube.rotation.y += 0.01;
    // controls.minAzimuthAngle += 0.01 * -Math.PI / 12; // left 45°

    const overlayVisible = isOverlayVisible();
    controls.autoRotate = overlayVisible;

    if (overlayVisible) {
        const azimuth = controls.getAzimuthalAngle();
        const epsilon = 0.001;
        
        if (azimuth >= controls.maxAzimuthAngle - epsilon) {
            controls.autoRotateSpeed = Math.abs(overlayAutoRotateSpeed);
        } else if (azimuth <= controls.minAzimuthAngle + epsilon) {
            controls.autoRotateSpeed = -Math.abs(overlayAutoRotateSpeed);
        }
    }

    controls.update(); // required when damping is on
    renderer.render(scene, camera);
}
animate();

scheduleOverlayReturn();

// Resize handling
window.addEventListener('resize', resizeRenderer);
