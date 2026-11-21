// embed.wgsl for a batch of tokens (prefill): x[t] is the bf16 row of ids[t],
// times SCALE. The embedding is one or two chunks; R0 is the first chunk's rows.
// An id at or past SLOT is an image token: its layer-0 state is slot[t], the
// projected vision row the host wrote for this batch (not scaled).

override D: u32;
override R0: u32;
override SCALE: f32 = 1.0;
override SLOT: u32 = 4294967295u;

@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> emb0: array<u32>;
@group(0) @binding(2) var<storage, read> emb1: array<u32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;   // [token][D]
@group(0) @binding(4) var<storage, read> nt: array<u32>;
@group(0) @binding(5) var<storage, read> slot: array<f32>;      // [token][D]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let t = gid.y;
  if (i >= D || t >= nt[0]) {
    return;
  }
  let id = ids[t];
  if (id >= SLOT) {
    x[t * D + i] = slot[t * D + i];
    return;
  }
  var w = 0u;
  var e = 0u;
  if (id < R0) {
    e = id * D + i;
    w = emb0[e >> 1u];
  } else {
    e = (id - R0) * D + i;
    w = emb1[e >> 1u];
  }
  x[t * D + i] = bitcast<f32>(select(w << 16u, w & 0xffff0000u, (e & 1u) == 1u)) * SCALE;
}
