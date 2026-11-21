// Causal attention for one new token: softmax(q k^T / sqrt(DENOM)) v over the
// cached positions 0..pos, one workgroup per query head. Grouped-query: head
// h reads key/value head h / (H / KVH).
//
// Every sum is the kernels directory's attention.wgsl's, in its order, so the
// output has its bits: a score is q.k added d = 0, 1, ...; the max and the
// exp sum are the same 256-thread trees; a value is att[t] v[t] added
// t = 0, 1, .... What changes is the loads: a key row is read as vec4s, and
// the value loop issues eight positions' loads before it adds them, in order,
// instead of waiting on each pair of loads in turn.

override H: u32;
override KVH: u32;
override HD: u32;     // a multiple of 4
override CTX: u32;
override DENOM: f32 = 0.0;

struct Step {
  pos: u32,
  row: u32,
  p0: u32,
  p1: u32,
}

@group(0) @binding(0) var<uniform> st: Step;
@group(0) @binding(1) var<storage, read> q: array<f32>;          // H * HD
@group(0) @binding(2) var<storage, read> kc: array<vec4<f32>>;   // CTX * KVH * HD / 4
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read_write> att: array<f32>;  // H * CTX scratch
@group(0) @binding(5) var<storage, read_write> o: array<f32>;    // H * HD

var<workgroup> red: array<f32, 256>;
var<workgroup> sq: array<vec4<f32>, 64>;  // q of this head, HD / 4 <= 64

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
  let h = wg.x;
  let kvh = h / (H / KVH);
  let n = st.pos + 1u;
  let qb = h * HD;
  let ab = h * CTX;
  let denom = select(f32(HD), DENOM, DENOM > 0.0);
  let scale = inverseSqrt(denom);
  let h4 = HD / 4u;
  for (var i = tid; i < h4; i += 256u) {
    sq[i] = vec4<f32>(q[qb + 4u * i], q[qb + 4u * i + 1u], q[qb + 4u * i + 2u], q[qb + 4u * i + 3u]);
  }
  workgroupBarrier();

  var m = -3.0e38;
  for (var t = tid; t < n; t += 256u) {
    let kb = (t * KVH + kvh) * h4;
    var s = 0.0;
    for (var d4 = 0u; d4 < h4; d4++) {
      let k = kc[kb + d4];
      let qq = sq[d4];
      s += qq.x * k.x;
      s += qq.y * k.y;
      s += qq.z * k.z;
      s += qq.w * k.w;
    }
    s *= scale;
    att[ab + t] = s;
    m = max(m, s);
  }
  let mx = reduce(m, tid, true);

  var sum = 0.0;
  for (var t = tid; t < n; t += 256u) {
    let e = exp(att[ab + t] - mx);
    att[ab + t] = e;
    sum += e;
  }
  let total = reduce(sum, tid, false);
  storageBarrier();

  let vs = KVH * HD;
  for (var d = tid; d < HD; d += 256u) {
    var acc = 0.0;
    let vb = kvh * HD + d;
    var t = 0u;
    for (; t + 8u <= n; t += 8u) {
      let a0 = att[ab + t];
      let a1 = att[ab + t + 1u];
      let a2 = att[ab + t + 2u];
      let a3 = att[ab + t + 3u];
      let a4 = att[ab + t + 4u];
      let a5 = att[ab + t + 5u];
      let a6 = att[ab + t + 6u];
      let a7 = att[ab + t + 7u];
      let v0 = vc[t * vs + vb];
      let v1 = vc[(t + 1u) * vs + vb];
      let v2 = vc[(t + 2u) * vs + vb];
      let v3 = vc[(t + 3u) * vs + vb];
      let v4 = vc[(t + 4u) * vs + vb];
      let v5 = vc[(t + 5u) * vs + vb];
      let v6 = vc[(t + 6u) * vs + vb];
      let v7 = vc[(t + 7u) * vs + vb];
      acc += a0 * v0;
      acc += a1 * v1;
      acc += a2 * v2;
      acc += a3 * v3;
      acc += a4 * v4;
      acc += a5 * v5;
      acc += a6 * v6;
      acc += a7 * v7;
    }
    for (; t < n; t++) {
      acc += att[ab + t] * vc[t * vs + vb];
    }
    o[qb + d] = acc / total;
  }
}
