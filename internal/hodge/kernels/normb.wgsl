// norm.wgsl for a batch of tokens (prefill): workgroup t normalizes token t.
// The arithmetic is norm.wgsl's line for line; only the indices carry the
// token's offset: a, b and y are [token][K], xq [token][K/4], xs [token].
//
//   MODE 0: v = a                      -> rmsnorm -> int8 (xq, xs) and f32 y
//   MODE 1: v = act(a) * b             -> rmsnorm -> int8 and f32 y
//   MODE 2: v = a                      -> rmsnorm -> f32 y

override K: u32;
override EPS: f32 = 1e-5;
override MODE: u32 = 0u;
override ACT: u32 = 0u;
override GEMMA: u32 = 0u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;          // MODE 1 only
@group(0) @binding(2) var<storage, read> nw: array<f32>;         // the norm's weight
@group(0) @binding(3) var<storage, read_write> xq: array<u32>;
@group(0) @binding(4) var<storage, read_write> xs: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

var<workgroup> red: array<f32, 256>;

fn v(t: u32, i: u32) -> f32 {
  if (MODE == 1u) {
    let x = a[t * K + i];
    if (ACT == 1u) {
      return x / (1.0 + exp(-x)) * b[t * K + i];
    }
    if (ACT == 2u) {
      return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))) * b[t * K + i];
    }
    let g = max(x, 0.0);
    return g * g * b[t * K + i];
  }
  return a[t * K + i];
}

fn sum256(x: f32, tid: u32) -> f32 {
  red[tid] = x;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      red[tid] += red[tid + s];
    }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

fn gain(i: u32, unit: bool) -> f32 {
  if (unit && MODE != 2u) {
    return 1.0;
  }
  if (GEMMA == 1u) {
    return 1.0 + nw[i];
  }
  return nw[i];
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
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) tid: u32) {
  let t = wg.x;
  var ss = 0.0;
  var ones = 0.0;
  for (var i = tid; i < K; i += 256u) {
    let x = v(t, i);
    ss += x * x;
    if (nw[i] == 1.0) {
      ones += 1.0;
    }
  }
  let invRms = inverseSqrt(sum256(ss, tid) / f32(K) + EPS);
  let unit = sum256(ones, tid) == f32(K);
  let inv = select(invRms, 1.0, unit && MODE != 2u);

  if (MODE == 2u) {
    for (var i = tid; i < K; i += 256u) {
      y[t * K + i] = v(t, i) * inv * gain(i, unit);
    }
    return;
  }

  var m = 0.0;
  for (var i = tid; i < K; i += 256u) {
    let z = v(t, i) * inv * gain(i, unit);
    y[t * K + i] = z;
    m = max(m, abs(z));
  }
  let s = 127.0 / max(max256(m, tid), 1e-5);
  for (var g = tid; g < K / 4u; g += 256u) {
    var word = 0u;
    for (var j = 0u; j < 4u; j++) {
      let i = g * 4u + j;
      let q = i32(clamp(round(v(t, i) * inv * gain(i, unit) * s), -128.0, 127.0));
      word |= (bitcast<u32>(q) & 0xffu) << (8u * j);
    }
    xq[t * (K / 4u) + g] = word;
  }
  if (tid == 0u) {
    xs[t] = 1.0 / s;
  }
}
