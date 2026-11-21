// rope.wgsl for a batch of tokens (prefill): gid.y is the token, at position
// st.pos + token. q, k, v are [token][heads * HD]. The arithmetic is
// rope.wgsl's line for line, so each token's cache rows are what one rope.wgsl
// dispatch at that position writes.

override H: u32;
override KVH: u32;
override HD: u32;
override THETA: f32;
override SCALING: u32 = 0u;
override FACTOR: f32 = 1.0;
override LOW_FF: f32 = 1.0;
override HIGH_FF: f32 = 4.0;
override ORIG: f32 = 8192.0;

struct Step {
  pos: u32,
  row: u32,
  p0: u32,
  p1: u32,
}

@group(0) @binding(0) var<uniform> st: Step;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;   // [token][H * HD]
@group(0) @binding(2) var<storage, read> k: array<f32>;         // [token][KVH * HD]
@group(0) @binding(3) var<storage, read> vv: array<f32>;        // [token][KVH * HD]
@group(0) @binding(4) var<storage, read_write> kc: array<f32>;  // ctx * KVH * HD
@group(0) @binding(5) var<storage, read_write> vc: array<f32>;
@group(0) @binding(6) var<storage, read> nt: array<u32>;        // [0]: tokens in the batch
@group(0) @binding(7) var<storage, read> ph: array<f32>;        // [token]: image token phase, 0 for text

fn nz(x: f32, fallback: f32) -> f32 {
  if (x == 0.0) {
    return fallback;
  }
  return x;
}

fn freq(d: u32) -> f32 {
  let f = pow(THETA, -2.0 * f32(d) / f32(HD));
  let factor = nz(FACTOR, 1.0);
  let low = nz(LOW_FF, 1.0);
  let high = nz(HIGH_FF, 4.0);
  let orig = nz(ORIG, 8192.0);
  if (SCALING == 1u) {
    return f / factor;
  }
  if (SCALING == 2u) {
    let wl = 6.283185307179586 / f;
    if (wl > orig / low) {
      return f / factor;
    }
    if (wl >= orig / high) {
      let span0 = high - low;
      let span = select(span0, 1.0, span0 == 0.0);
      let sm = (orig / wl - low) / span;
      return (1.0 - sm) * f / factor + sm * f;
    }
  }
  return f;
}

fn rot(x1: f32, x2: f32, d: u32, pos: u32, t: u32) -> vec2<f32> {
  let inv = freq(d);
  var ang = f32(pos) * inv;
  if (d >= 56u && d < 64u) {
    ang += bitcast<f32>(st.p0) + ph[t];
  }
  let c = cos(ang);
  let s = sin(ang);
  return vec2<f32>(x1 * c - x2 * s, x2 * c + x1 * s);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.y;
  if (t >= nt[0]) {
    return;
  }
  let pos = st.pos + t;
  let half = HD / 2u;
  let idx = gid.x;
  let nq = H * half;
  let qo = t * H * HD;
  let ko = t * KVH * HD;
  if (idx < nq) {
    let base = qo + (idx / half) * HD;
    let d = idx % half;
    let r = rot(q[base + d], q[base + d + half], d, pos, t);
    q[base + d] = r.x;
    q[base + d + half] = r.y;
    return;
  }
  let j = idx - nq;
  if (j >= KVH * half) {
    return;
  }
  let base = (j / half) * HD;
  let d = j % half;
  let r = rot(k[ko + base + d], k[ko + base + d + half], d, pos, t);
  let at = pos * KVH * HD + base;
  kc[at + d] = r.x;
  kc[at + d + half] = r.y;
  vc[at + d] = vv[ko + base + d];
  vc[at + d + half] = vv[ko + base + d + half];
}
