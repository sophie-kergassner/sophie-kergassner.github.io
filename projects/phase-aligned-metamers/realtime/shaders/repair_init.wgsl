// Levels that are not repaired (>= cutoff): the repaired statistics are the measured ones
// (Worm.add_repaired_layer_proxies), with intensity = |G + iH| * intensity modulator.
// Requires sampling.wgsl, repair_common.wgsl.

@group(0) @binding(1) var ana: texture_2d<f32>;                          // dir, phase, |G + iH|, S
@group(0) @binding(2) var rep: texture_storage_2d<rgba32float, write>;   // intensity_rep, dir_rep, phase_rep, 0

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(rep);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2<i32>(id.xy);
    let a = textureLoad(ana, p, 0);
    textureStore(rep, p, vec4<f32>(a.z * modulator(), a.x, a.y, 0.0));
}
