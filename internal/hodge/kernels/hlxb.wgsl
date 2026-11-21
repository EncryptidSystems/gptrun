// hlx.wgsl for a batch of tokens (prefill): each weight word is read once per
// T tokens instead of once per token. Needs dot4 prepended.
//
// Activations are [token][K/4] int8 words (normb.wgsl), outputs [token][4*np]
// per matrix. For every token the float math is hlx.wgsl's:
//   f = vec4(i0) * (xs[t] * s0);  f = f + vec4(i1) * (xs[t] * s1);  ...
// and with RESID, o0 = o0 + f. The integer dot is exact in any order, so the
// lane sums are added by one thread per (row, token) from workgroup memory.

override K: u32;              // inputs
override NM: u32 = 1u;        // matrices in this dispatch, 1..3
override MAXL: u32 = 1u;      // most limbs any matrix here has
override RESID: bool = false; // o0 += f (NM is 1)

const ROWS = 8u;   // packed rows per workgroup
const LANES = 32u; // threads per packed row
const T = 8u;      // tokens per tile

struct Mat {
  start: u32,
  np: u32,
  nl: u32,
  pad: u32,
  woff: array<u32, 8>,
  center: array<i32, 8>,
  scale: array<f32, 8>,
}

@group(0) @binding(0) var<storage, read> xq: array<u32>;   // [token][K/4]
@group(0) @binding(1) var<storage, read> xs: array<f32>;   // [token]
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> mats: array<Mat>;
@group(0) @binding(4) var<storage, read> nt: array<u32>;   // [0]: tokens in this batch
@group(0) @binding(5) var<storage, read_write> o0: array<f32>;
@group(0) @binding(6) var<storage, read_write> o1: array<f32>;
@group(0) @binding(7) var<storage, read_write> o2: array<f32>;

// Lane partial sums for half the tile at a time: red[t * 256 + tid].
var<workgroup> red: array<vec4<i32>, 1024>;  // T/2 * 256

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) tid: u32) {
  let kw = K / 4u;
  let lane = tid % LANES;
  let slot = tid / LANES;
  let g = (wg.y * nwg.x + wg.x) * ROWS + slot;
  var m = 0u;
  for (var i = 1u; i < NM; i++) {
    if (g >= mats[i].start) {
      m = i;
    }
  }
  let mt = mats[m];
  let r = g - mt.start;
  let valid = r < mt.np;
  let n = nt[0];
  let so = tid / T;
  let to = tid % T;
  let sg = (wg.y * nwg.x + wg.x) * ROWS + so;
  var sm = 0u;
  for (var i = 1u; i < NM; i++) {
    if (sg >= mats[i].start) {
      sm = i;
    }
  }
  let smt = mats[sm];
  let sr = sg - smt.start;
  let svalid = tid < ROWS * T && sr < smt.np;

  for (var t0 = 0u; t0 < n; t0 += T) {
    var f = vec4<f32>(0.0);
    for (var l = 0u; l < MAXL; l++) {
      let live = valid && l < mt.nl;
      var acc: array<vec4<i32>, 8>;
      for (var t = 0u; t < T; t++) {
        acc[t] = vec4<i32>(0);
      }
      if (live) {
        let base = mt.woff[l] + r * kw;
        let c = mt.center[l];
        for (var k = lane; k < kw; k += LANES) {
          let ww = w[base + k];
          let c0 = ww & 0x03030303u;
          let c1 = (ww >> 2u) & 0x03030303u;
          let c2 = (ww >> 4u) & 0x03030303u;
          let c3 = (ww >> 6u) & 0x03030303u;
          for (var t = 0u; t < T; t++) {
            if (t0 + t < n) {
              let a = xq[(t0 + t) * kw + k];
              let ones = dot4(a, 0x01010101u);
              acc[t] += vec4<i32>(dot4(a, c0), dot4(a, c1), dot4(a, c2), dot4(a, c3)) - vec4<i32>(ones * c);
            }
          }
        }
      }
      for (var h = 0u; h < 2u; h++) {
        for (var t = 0u; t < T / 2u; t++) {
          red[t * 256u + tid] = acc[h * (T / 2u) + t];
        }
        workgroupBarrier();
        if (svalid && l < smt.nl && to / (T / 2u) == h && t0 + to < n) {
          var s = vec4<i32>(0);
          let at = (to % (T / 2u)) * 256u + so * LANES;
          for (var j = 0u; j < LANES; j++) {
            s += red[at + j];
          }
          let v = vec4<f32>(s) * (xs[t0 + to] * smt.scale[l]);
          if (l == 0u) {
            f = v;
          } else {
            f = f + v;
          }
        }
        workgroupBarrier();
      }
    }
    if (svalid && t0 + to < n) {
      let ob = (t0 + to) * 4u * smt.np;
      for (var i = 0u; i < 4u; i++) {
        let idx = ob + i * smt.np + sr;
        if (RESID) {
          o0[idx] = o0[idx] + f[i];
        } else if (sm == 0u) {
          o0[idx] = f[i];
        } else if (sm == 1u) {
          o1[idx] = f[i];
        } else {
          o2[idx] = f[i];
        }
      }
    }
  }
}
