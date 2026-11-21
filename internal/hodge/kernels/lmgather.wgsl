// lmhead.wgsl for a list of rows: the exact bf16 logit of each row the screen
// kept. The arithmetic is lmhead.wgsl's line for line (same staging, same lane
// order, same tree), so each value is the bit pattern lmhead.wgsl writes.
// cand[CB + i] is a row of this chunk; cnt[CI] is how many there are.

override D: u32;      // hidden size (even)
override CB: u32;     // this chunk's first slot in cand and ex
override CI: u32;     // this chunk's index in cnt

const PER = 8u;       // rows per workgroup
const LANES = 32u;

@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> emb: array<u32>;          // this chunk, rows x D bf16
@group(0) @binding(2) var<storage, read> cand: array<u32>;
@group(0) @binding(3) var<storage, read> cnt: array<u32>;
@group(0) @binding(4) var<storage, read_write> ex: array<f32>;

var<workgroup> sy: array<f32, D>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) tid: u32) {
  for (var i = tid; i < D; i += 256u) {
    sy[i] = y[i];
  }
  workgroupBarrier();
  let lane = tid % LANES;
  let slot = (wg.y * nwg.x + wg.x) * PER + tid / LANES;
  let n = cnt[CI];
  var acc = 0.0;
  if (slot < n) {
    let r = cand[CB + slot];
    let base = r * (D / 2u);
    for (var k = lane; k < D / 2u; k += LANES) {
      let w = emb[base + k];
      acc += sy[2u * k] * bitcast<f32>(w << 16u) + sy[2u * k + 1u] * bitcast<f32>(w & 0xffff0000u);
    }
  }
  red[tid] = acc;
  workgroupBarrier();
  for (var s = LANES / 2u; s > 0u; s >>= 1u) {
    if (lane < s) {
      red[tid] += red[tid + s];
    }
    workgroupBarrier();
  }
  if (lane == 0u && slot < n) {
    ex[CB + slot] = red[tid];
  }
}
