// One level of the Gaussian pyramid, as isu.get_gauss_mipmaps (gamma=None):
// 5x5 Gaussian (sigma 1, normalized; weights from constants.json), torch 'reflect' padding, then every 2nd pixel.
// Requires sampling.wgsl.

@group(0) @binding(0) var<storage, read> kern: array<f32, 25>;
@group(0) @binding(1) var src: texture_2d<f32>;                          // finer level (rgba32float, YCbCr)
@group(0) @binding(2) var dst: texture_storage_2d<rgba32float, write>;   // coarser level, ceil(size / 2)

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let n = vec2<i32>(textureDimensions(src));
    let c = vec2<i32>(id.xy) * 2;
    var acc = vec4<f32>(0.0);
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let q = vec2<i32>(reflect_torch(c.x + dx, n.x), reflect_torch(c.y + dy, n.y));
            acc += kern[(dy + 2) * 5 + (dx + 2)] * textureLoad(src, q, 0);
        }
    }
    textureStore(dst, vec2<i32>(id.xy), acc);
}
