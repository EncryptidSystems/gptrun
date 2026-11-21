// The LM head screen: ap[r] = s_r * sum_j y_j q_rj over the int8 copy of the
// embedding (lmquant.wgsl). Half the bytes of lmhead.wgsl. The host bounds
// |ap - logit| by ||y||_1 * (d_r + f32 slack) and rescores, exactly, only the
// rows that bound cannot rule out (lmgather.wgsl). The order of this sum is
// free: the bound holds for any order.
//
// 32 lanes per row, each loading 16 int8 at a time; a workgroup walks PASSES
// groups of 8 rows so y is staged once per 8*PASSES rows. y is staged
// transposed, sy[i * D/16 + k] = y[16k + 4i .. 16k + 4i + 3], so the lanes of a
// row read consecutive vec4s.

override D: u32;     // hidden size, a multiple of 16
override ROWS: u32;  // vocabulary

const PER = 8u;      // rows per pass
const PASSES = 8u;   // passes per workgroup
const LANES = 32u;

@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> q8: array<vec4<u32>>;   // ROWS x D/16
@group(0) @binding(2) var<storage, read> sd: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> ap: array<f32>;

var<workgroup> sy: array<vec4<f32>, D / 4u>;
var<workgroup> red: array<f32, 256>;

fn unpack(w: u32) -> vec4<f32> {
  return vec4<f32>(f32(bitcast<i32>(w << 24u) >> 24u), f32(bitcast<i32>(w << 16u) >> 24u),
                   f32(bitcast<i32>(w << 8u) >> 24u), f32(bitcast<i32>(w) >> 24u));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) tid: u32) {
  let kq = D / 16u;
  for (var wi = tid; wi < D / 4u; wi += 256u) {
    let k = wi / 4u;
    let i = wi % 4u;
    sy[i * kq + k] = vec4<f32>(y[4u * wi], y[4u * wi + 1u], y[4u * wi + 2u], y[4u * wi + 3u]);
  }
  workgroupBarrier();
  let lane = tid % LANES;
  let first = (wg.y * nwg.x + wg.x) * PER * PASSES + tid / LANES;
  for (var p = 0u; p < PASSES; p++) {
    let r = first + p * PER;
    var acc = 0.0;
    if (r < ROWS) {
      let base = r * kq;
      for (var k = lane; k < kq; k += LANES) {
        let v = q8[base + k];
        acc += dot(sy[k], unpack(v.x)) + dot(sy[kq + k], unpack(v.y)) +
               dot(sy[2u * kq + k], unpack(v.z)) + dot(sy[3u * kq + k], unpack(v.w));
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
    if (lane == 0u && r < ROWS) {
      ap[r] = red[tid] * sd[r].x;
    }
    workgroupBarrier();
  }
}
