// HodgeLinear, fused: up to three packed ternary matrices that share one input
// (q, k, v or gate, up), every digit limb of each, and optionally the residual
// add, in one dispatch. Needs dot4 (dot4_native.wgsl or dot4_soft.wgsl) prepended.
//
// The weight layout is hodgelinear.wgsl's: byte k of packed row r holds the
// 2-bit codes of output rows r, r + NP, r + 2NP, r + 3NP at input k. All limbs
// of all matrices sit in one buffer w; meta[m] gives each limb's word offset,
// CENTER and scale.
//
// The float math is hodgelinear.wgsl's, limb by limb in limb order:
//   f = vec4(i0) * (xs * s0);  f = f + vec4(i1) * (xs * s1);  ...
// which is what limb 0 writing and limbs 1.. ACCUM-adding through memory
// computed. The integer dot is exact in any order. With RESID the result is
// added to o0 (the residual stream): x = x + f, as the separate add did.

override K: u32;              // inputs
override NM: u32 = 1u;        // matrices in this dispatch, 1..3
override MAXL: u32 = 1u;      // most limbs any matrix here has, 1..8
override RESID: bool = false; // o0 += f instead of o = f (NM is 1)

const ROWS = 8u;   // packed rows per workgroup
const LANES = 32u; // threads per packed row

struct Mat {
  start: u32,          // first packed row of this matrix in the dispatch
  np: u32,             // packed rows = outputs / 4
  nl: u32,             // limbs
  pad: u32,
  woff: array<u32, 8>,   // word offset of each limb in w
  center: array<i32, 8>, // the code that means 0, per limb
  scale: array<f32, 8>,  // weight scale, per limb
}

@group(0) @binding(0) var<storage, read> xq: array<u32>;   // K/4 words of int8
@group(0) @binding(1) var<storage, read> xs: array<f32>;   // [0]: activation dequant scale
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> mats: array<Mat>;
@group(0) @binding(4) var<storage, read_write> o0: array<f32>;
@group(0) @binding(5) var<storage, read_write> o1: array<f32>;
@group(0) @binding(6) var<storage, read_write> o2: array<f32>;

var<workgroup> sx: array<u32, K / 4u>;
var<workgroup> red: array<vec4<i32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) tid: u32) {
  let kw = K / 4u;
  for (var k = tid; k < kw; k += 256u) {
    sx[k] = xq[k];
  }
  workgroupBarrier();

  let lane = tid % LANES;
  let g = (wg.y * nwg.x + wg.x) * ROWS + tid / LANES;
  var m = 0u;
  for (var i = 1u; i < NM; i++) {
    if (g >= mats[i].start) {
      m = i;
    }
  }
  let mt = mats[m];
  let r = g - mt.start;
  let valid = r < mt.np;
  let xsv = xs[0];

  var f = vec4<f32>(0.0);
  for (var l = 0u; l < MAXL; l++) {
    let live = valid && l < mt.nl;
    var acc = vec4<i32>(0);
    if (live) {
      let base = mt.woff[l] + r * kw;
      let c = mt.center[l];
      for (var k = lane; k < kw; k += LANES) {
        let ww = w[base + k];
        let a = sx[k];
        let ones = dot4(a, 0x01010101u);
        acc += vec4<i32>(dot4(a, ww & 0x03030303u),
                         dot4(a, (ww >> 2u) & 0x03030303u),
                         dot4(a, (ww >> 4u) & 0x03030303u),
                         dot4(a, (ww >> 6u) & 0x03030303u)) - vec4<i32>(ones * c);
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
    if (lane == 0u && live) {
      let v = vec4<f32>(red[tid]) * (xsv * mt.scale[l]);
      if (l == 0u) {
        f = v;
      } else {
        f = f + v;
      }
    }
    workgroupBarrier();
  }

  if (lane == 0u && valid) {
    for (var i = 0u; i < 4u; i++) {
      let idx = i * mt.np + r;
      if (RESID) {
        o0[idx] = o0[idx] + f[i];
      } else if (m == 0u) {
        o0[idx] = f[i];
      } else if (m == 1u) {
        o1[idx] = f[i];
      } else {
        o2[idx] = f[i];
      }
    }
  }
}
