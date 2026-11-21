// One Jacobi sweep of the eikonal field on a CSR graph, every vertex from the
// previous sweep: the fast-sweeping / Bellman–Ford form the HES note gives
// for parallel hardware.
//
// forward:  T'(v) = softmin_β( S(v), T(u) + w_uv over neighbours u )
//           (HARD = 1: the plain min, the hard field)
// adjoint:  μ'(v) = dL(v) + Σ_u e^(−β(T(v) + w_vu − T(u))) μ(u)
//           the transpose of the soft map at its fixed point T.
//
// dlt[v] is |new − old| for the convergence test.

struct P {
  n: u32,
  hard: u32,
  beta: f32,
  pad: f32,
};

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> off: array<u32>;
@group(0) @binding(2) var<storage, read> nbr: array<u32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;      // per adjacency entry
@group(0) @binding(4) var<storage, read> src: array<f32>;    // S (forward) or dL (adjoint)
@group(0) @binding(5) var<storage, read> tin: array<f32>;
@group(0) @binding(6) var<storage, read_write> tout: array<f32>;
@group(0) @binding(7) var<storage, read_write> dlt: array<f32>;
@group(0) @binding(8) var<storage, read> tf: array<f32>;     // converged T (adjoint)

const INF: f32 = 3.0e38;

fn smin(beta: f32, a: f32, b: f32) -> f32 {
  if (a >= INF) {
    return b;
  }
  if (b >= INF) {
    return a;
  }
  let m = min(a, b);
  return m - log(exp(-beta * (a - m)) + exp(-beta * (b - m))) / beta;
}

@compute @workgroup_size(256)
fn forward(@builtin(global_invocation_id) gid: vec3u) {
  let v = gid.x;
  if (v >= p.n) {
    return;
  }
  var x = src[v];
  let end = off[v + 1u];
  for (var i = off[v]; i < end; i++) {
    let t = tin[nbr[i]];
    if (t >= INF) {
      continue;
    }
    let c = t + w[i];
    if (p.hard == 1u) {
      x = min(x, c);
    } else {
      x = smin(p.beta, x, c);
    }
  }
  tout[v] = x;
  let old = tin[v];
  if (x >= INF) {
    dlt[v] = 0.0;
  } else if (old >= INF) {
    dlt[v] = INF;
  } else {
    dlt[v] = abs(x - old);
  }
}

@compute @workgroup_size(256)
fn adjoint(@builtin(global_invocation_id) gid: vec3u) {
  let v = gid.x;
  if (v >= p.n) {
    return;
  }
  var x = src[v];
  let tv = tf[v];
  if (tv < INF) {
    let end = off[v + 1u];
    for (var i = off[v]; i < end; i++) {
      let u = nbr[i];
      let tu = tf[u];
      if (tu >= INF) {
        continue;
      }
      x = x + exp(-p.beta * (tv + w[i] - tu)) * tin[u];
    }
  }
  tout[v] = x;
  dlt[v] = abs(x - tin[v]);
}
