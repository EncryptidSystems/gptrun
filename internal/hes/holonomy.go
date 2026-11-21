package hes

import "math"

// Channel 1 of the HES note is a running product over a group: S_n as
// permutation matrices, SO(n) or U(n). perm.go is S_n. Rot is the orthogonal
// channel, built the DeltaProduct way: each step multiplies two generalized
// Householder reflections I − 2vvᵀ, so every step is in SO(n) and the product
// does not commute. Concatenating two paths multiplies their states (Chen).

// Inv is the inverse permutation.
func (g Perm) Inv() Perm {
	out := make(Perm, len(g))
	for i, j := range g {
		if j >= 0 && j < len(g) {
			out[j] = i
		}
	}
	return out
}

// IsIdentity reports g = e.
func (g Perm) IsIdentity() bool {
	for i, j := range g {
		if i != j {
			return false
		}
	}
	return true
}

// Rot is an n×n orthogonal matrix, row major.
type Rot struct {
	N int
	A []float64
}

// RotIdentity is I_n.
func RotIdentity(n int) Rot {
	r := Rot{N: n, A: make([]float64, n*n)}
	for i := 0; i < n; i++ {
		r.A[i*n+i] = 1
	}
	return r
}

// Clone copies r.
func (r Rot) Clone() Rot { return Rot{N: r.N, A: append([]float64(nil), r.A...)} }

// Mul is r·t.
func (r Rot) Mul(t Rot) Rot {
	n := r.N
	out := Rot{N: n, A: make([]float64, n*n)}
	for i := 0; i < n; i++ {
		for k := 0; k < n; k++ {
			a := r.A[i*n+k]
			if a == 0 {
				continue
			}
			for j := 0; j < n; j++ {
				out.A[i*n+j] += a * t.A[k*n+j]
			}
		}
	}
	return out
}

// reflectRight is r·(I − 2vvᵀ) for a unit v, in place.
func (r *Rot) reflectRight(v []float64) {
	n := r.N
	for i := 0; i < n; i++ {
		var dot float64
		row := r.A[i*n : i*n+n]
		for j := 0; j < n; j++ {
			dot += row[j] * v[j]
		}
		dot *= 2
		for j := 0; j < n; j++ {
			row[j] -= dot * v[j]
		}
	}
}

// Householder2 is the step T(x) = (I − 2aaᵀ)(I − 2bbᵀ) for the two directions
// a, b (normalized here). A zero direction is skipped.
func Householder2(n int, a, b []float64) Rot {
	t := RotIdentity(n)
	for _, v := range [][]float64{a, b} {
		u := unit(v, n)
		if u != nil {
			t.reflectRight(u)
		}
	}
	return t
}

func unit(v []float64, n int) []float64 {
	if len(v) < n {
		return nil
	}
	var ss float64
	for i := 0; i < n; i++ {
		ss += v[i] * v[i]
	}
	if ss == 0 {
		return nil
	}
	inv := 1 / math.Sqrt(ss)
	u := make([]float64, n)
	for i := 0; i < n; i++ {
		u[i] = v[i] * inv
	}
	return u
}

// Distance is the Frobenius distance ‖r − t‖. Two developments of the same
// path agree exactly; a different order of the same steps does not.
func (r Rot) Distance(t Rot) float64 {
	var ss float64
	for i := range r.A {
		d := r.A[i] - t.A[i]
		ss += d * d
	}
	return math.Sqrt(ss)
}

// Trace is tr(r). tr = n exactly at the identity.
func (r Rot) Trace() float64 {
	var s float64
	for i := 0; i < r.N; i++ {
		s += r.A[i*r.N+i]
	}
	return s
}
