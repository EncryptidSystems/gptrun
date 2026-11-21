// Once, at load: each embedding row (bf16) to int8 with one scale per row, for
// the LM head screen. sd[r] = (s, d): s = max|w| / 127, and d = max |w - q s|,
// the row's largest rounding error, which the screen's error bound uses.
// One workgroup per row of this chunk.

override D: u32;     // hidden size, a multiple of 4
override ROW0: u32;  // this chunk's first row in the whole vocabulary
override ROWS: u32;  // rows in this chunk

@group(0) @binding(0) var<storage, read> emb: array<u32>;          // ROWS x D bf16
@group(0) @binding(1) var<storage, read_write> q8: array<u32>;     // vocab x D/4 words of int8
@group(0) @binding(2) var<storage, read_write> sd: array<vec2<f32>>;

var<workgroup> red: array<f32, 256>;

fn wv(r: u32, j: u32) -> f32 {
  let e = r * D + j;
  let w = emb[e >> 1u];
  return bitcast<f32>(select(w << 16u, w & 0xffff0000u, (e & 1u) == 1u));
}

fn max256(x: f32, tid: u32) -> f32 {
  red[tid] = x;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      red[tid] = max(red[tid], red[tid + s]);
    }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) tid: u32) {
  let lr = wg.y * nwg.x + wg.x;
  if (lr >= ROWS) {
    return;
  }
  var m = 0.0;
  for (var j = tid; j < D; j += 256u) {
    m = max(m, abs(wv(lr, j)));
  }
  let s = max256(m, tid) / 127.0;
  let row = ROW0 + lr;
  var dmax = 0.0;
  for (var g = tid; g < D / 4u; g += 256u) {
    var word = 0u;
    for (var i = 0u; i < 4u; i++) {
      let w = wv(lr, 4u * g + i);
      var q = 0.0;
      if (s > 0.0) {
        q = clamp(round(w / s), -127.0, 127.0);
      }
      dmax = max(dmax, abs(w - q * s));
      word |= (bitcast<u32>(i32(q)) & 0xffu) << (8u * i);
    }
    q8[row * (D / 4u) + g] = word;
  }
  let d = max256(dmax, tid);
  if (tid == 0u) {
    sd[row] = vec2<f32>(s, d);
  }
}
