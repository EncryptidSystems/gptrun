// The residual add: x += w. The same sum, element by element, that wedge.wgsl
// did on one thread; its flag (the cell area) was never read.

override N: u32;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i < N) {
    x[i] = x[i] + w[i];
  }
}
