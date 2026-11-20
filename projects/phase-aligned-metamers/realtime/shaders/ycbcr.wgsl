// RGB -> YCbCr, identical to isu.rgb_2_ycrcb:
//   Y  = 0.299 R + 0.587 G + 0.114 B
//   c1 = 0.5 + 0.713 (R - Y)        (stored in .g)
//   c2 = 0.5 + 0.564 (B - Y)        (stored in .b)

@group(0) @binding(0) var src: texture_2d<f32>;                          // rgba8unorm input image
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;   // YCbCr

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    let c = textureLoad(src, p, 0).rgb;
    let y = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
    textureStore(dst, p, vec4<f32>(y, 0.5 + 0.713 * (c.r - y), 0.5 + 0.564 * (c.b - y), 1.0));
}
