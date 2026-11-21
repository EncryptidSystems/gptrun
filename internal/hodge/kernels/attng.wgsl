// attn.wgsl for a whole grouped-query group: workgroup g is key/value head g
// and its G = H / KVH query heads, so each cached key and value row is read
// once for the group instead of once per query head. For every head the sums
// are attention.wgsl's, in its order (a score adds d = 0, 1, ...; the max and
// the exp sum are the same 256-thread trees; a value adds t = 0, 1, ...), so
// each head's output has its bits.

override H: u32;
override KVH: u32;
override HD: u32;     // a multiple of 4, at most 256
override CTX: u32;
override DENOM: f32 = 0.0;

const MAXG = 8u;      // query heads per group this kernel holds

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
var<workgroup> sq: array<vec4<f32>, 512>;  // the group's q, MAXG * HD / 4

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
  let G = H / KVH;
  let kvh = wg.x;
  let h0 = kvh * G;
  let n = st.pos + 1u;
  let denom = select(f32(HD), DENOM, DENOM > 0.0);
  let scale = inverseSqrt(denom);
  let h4 = HD / 4u;
  for (var i = tid; i < G * h4; i += 256u) {
    let qb = h0 * HD + 4u * i;
    sq[i] = vec4<f32>(q[qb], q[qb + 1u], q[qb + 2u], q[qb + 3u]);
  }
  workgroupBarrier();

  var m: array<f32, MAXG>;
  for (var j = 0u; j < G; j++) {
    m[j] = -3.0e38;
  }
  for (var t = tid; t < n; t += 256u) {
    let kb = (t * KVH + kvh) * h4;
    var s: array<f32, MAXG>;
    for (var j = 0u; j < G; j++) {
      s[j] = 0.0;
    }
    for (var d4 = 0u; d4 < h4; d4++) {
      let k = kc[kb + d4];
      for (var j = 0u; j < G; j++) {
        let qq = sq[j * h4 + d4];
        s[j] += qq.x * k.x;
        s[j] += qq.y * k.y;
        s[j] += qq.z * k.z;
        s[j] += qq.w * k.w;
      }
    }
    for (var j = 0u; j < G; j++) {
      let v = s[j] * scale;
      att[(h0 + j) * CTX + t] = v;
      m[j] = max(m[j], v);
    }
  }
  var mx: array<f32, MAXG>;
  for (var j = 0u; j < G; j++) {
    mx[j] = reduce(m[j], tid, true);
  }

  var sum: array<f32, MAXG>;
  for (var j = 0u; j < G; j++) {
    sum[j] = 0.0;
  }
  for (var t = tid; t < n; t += 256u) {
    for (var j = 0u; j < G; j++) {
      let at = (h0 + j) * CTX + t;
      let e = exp(att[at] - mx[j]);
      att[at] = e;
      sum[j] += e;
    }
  }
  var total: array<f32, MAXG>;
  for (var j = 0u; j < G; j++) {
    total[j] = reduce(sum[j], tid, false);
  }
  storageBarrier();

  let vs = KVH * HD;
  for (var d = tid; d < HD; d += 256u) {
    var acc: array<f32, MAXG>;
    for (var j = 0u; j < G; j++) {
      acc[j] = 0.0;
    }
    let vb = kvh * HD + d;
    var t = 0u;
    for (; t + 4u <= n; t += 4u) {
      let v0 = vc[t * vs + vb];
      let v1 = vc[(t + 1u) * vs + vb];
      let v2 = vc[(t + 2u) * vs + vb];
      let v3 = vc[(t + 3u) * vs + vb];
      for (var j = 0u; j < G; j++) {
        let ab = (h0 + j) * CTX + t;
        let a0 = att[ab];
        let a1 = att[ab + 1u];
        let a2 = att[ab + 2u];
        let a3 = att[ab + 3u];
        acc[j] += a0 * v0;
        acc[j] += a1 * v1;
        acc[j] += a2 * v2;
        acc[j] += a3 * v3;
      }
    }
    for (; t < n; t++) {
      let v = vc[t * vs + vb];
      for (var j = 0u; j < G; j++) {
        acc[j] += att[(h0 + j) * CTX + t] * v;
      }
    }
    for (var j = 0u; j < G; j++) {
      o[(h0 + j) * HD + d] = acc[j] / total[j];
    }
  }
}
