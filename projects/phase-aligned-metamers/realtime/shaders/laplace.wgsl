// Laplacian band of level i (Y channel), as Worm.compute_laplace_layers:
//   laplace_i = Y_i - isu.upsample(Y_{i+1})
// Requires sampling.wgsl.

@group(0) @binding(0) var fine: texture_2d<f32>;                        // level i (rgba32float, YCbCr)
@group(0) @binding(1) var coarse: texture_2d<f32>;                      // level i+1
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;     // Laplacian band at the size of level i

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    let up = upsample_at(coarse, p, 2.0);
    textureStore(dst, p, vec4<f32>(textureLoad(fine, p, 0).r - up.r, 0.0, 0.0, 0.0));
}
