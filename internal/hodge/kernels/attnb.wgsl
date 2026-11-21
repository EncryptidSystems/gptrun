// attention.wgsl for a batch of tokens (prefill): workgroup (h, t) is query
// head h of token t, which sees positions 0 .. st.pos + t. The arithmetic is
// attention.wgsl's line for line. q and o are [token][H * HD]; att is
// [token][H][CTX] scratch.

override H: u32;
override KVH: u32;
override HD: u32;
override CTX: u32;
override DENOM: f32 = 0.0;

struct Step {
  pos: u32,
  row: u32,
  p0: u32,
  p1: u32,
}

@group(0) @binding(0) var<uniform> st: Step;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<f32>;
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read_write> att: array<f32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
@group(0) @binding(6) var<storage, read> nt: array<u32>;

var<workgroup> red: array<f32, 256>;

fn reduce(x: f32, tid: u32, is_max: bool) -> f32 {
  red[tid] = x;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      if (is_max) {
        red[tid] = max(red[tid], red[tid + s]);
      } else {
        red[tid] += red[tid + s];
      }
    }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) tid: u32) {
  let t = wg.y;
  if (t >= nt[0]) {
    return;
  }
  let h = wg.x;
  let kvh = h / (H / KVH);
  let n = st.pos + t + 1u;
  let qb = (t * H + h) * HD;
  let ab = (t * H + h) * CTX;
  let denom = select(f32(HD), DENOM, DENOM > 0.0);
  let scale = inverseSqrt(denom);

  var m = -3.0e38;
  for (var u = tid; u < n; u += 256u) {
    let kb = (u * KVH + kvh) * HD;
    var s = 0.0;
    for (var d = 0u; d < HD; d++) {
      s += q[qb + d] * kc[kb + d];
    }
    s *= scale;
    att[ab + u] = s;
    m = max(m, s);
  }
  let mx = reduce(m, tid, true);

  var sum = 0.0;
  for (var u = tid; u < n; u += 256u) {
    let e = exp(att[ab + u] - mx);
    att[ab + u] = e;
    sum += e;
  }
  let total = reduce(sum, tid, false);
  storageBarrier();

  for (var d = tid; d < HD; d += 256u) {
    var acc = 0.0;
    for (var u = 0u; u < n; u++) {
      acc += att[ab + u] * vc[(u * KVH + kvh) * HD + d];
    }
    o[qb + d] = acc / total;
  }
}
