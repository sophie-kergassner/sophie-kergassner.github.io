// Moving input (temporal stability test): the image translated by a whole number of pixels, so that it is exactly
// the original content, shifted (no resampling blur that would itself change from frame to frame).
// Edges: the border pixel is repeated.

struct ShiftParams {
    offset: vec4<f32>,   // (dx, dy) [px], integers; content moves right / down for positive values
}

@group(0) @binding(0) var<uniform> SP: ShiftParams;
@group(0) @binding(1) var src: texture_2d<f32>;                          // YCbCr of the loaded image
@group(0) @binding(2) var dst: texture_storage_2d<rgba32float, write>;   // YCbCr input of the pipeline

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    let q = clamp(p - vec2<i32>(SP.offset.xy), vec2<i32>(0), vec2<i32>(dims) - 1);
    textureStore(dst, p, textureLoad(src, q, 0));
}
