package hes

import "fmt"

// S5 is the permutation group the holonomy channel is built to track.
// Adjacent transpositions of five letters do not commute, so a product can
// separate words a diagonal recurrence cannot.
const S5 = 5

// OneHop is the smallest key window in which a write can improve a neighbor
// and then stop. The march does not run past this window.
const OneHop = 1

// Perm is a permutation of Rank letters, stored as the image of 0..n-1.
// Multiplication is the path order g · T: T is applied first.
type Perm []int

// Identity is the identity in S_n.
func Identity(n int) Perm {
	p := make(Perm, n)
	for i := range p {
		p[i] = i
	}
	return p
}

// Adjacent is the transposition (i i+1). i is taken modulo n-1.
func Adjacent(n, i int) Perm {
	p := Identity(n)
	if n < 2 {
		return p
	}
	i %= n - 1
	if i < 0 {
		i += n - 1
	}
	p[i], p[i+1] = p[i+1], p[i]
	return p
}

// Transposition swaps a and b.
func Transposition(n, a, b int) Perm {
	p := Identity(n)
	if n == 0 {
		return p
	}
	a %= n
	b %= n
	if a < 0 {
		a += n
	}
	if b < 0 {
		b += n
	}
	p[a], p[b] = p[b], p[a]
	return p
}

// FromToken maps a token id into S_n as a word in the adjacent transpositions.
// The base-(n-1) digits of id are the generators, least digit applied first.
// Id 0 is the identity. The generators of S_n are these transpositions, so a
// long enough word can be any permutation; a single token is one such word.
func FromToken(n, id int) Perm {
	g := Identity(n)
	if n < 2 || id == 0 {
		return g
	}
	if id < 0 {
		id = -id
	}
	base := n - 1
	for id > 0 {
		d := id % base
		id /= base
		g = g.Mul(Adjacent(n, d))
	}
	return g
}

// Mul is g · t. The product applies t first.
func (g Perm) Mul(t Perm) Perm {
	n := len(g)
	if len(t) < n {
		n = len(t)
	}
	out := Identity(len(g))
	for i := 0; i < n; i++ {
		j := t[i]
		if j < 0 || j >= len(g) {
			j = i
		}
		out[i] = g[j]
	}
	return out
}

// Eq reports whether the two permutations agree.
func (g Perm) Eq(t Perm) bool {
	if len(g) != len(t) {
		return false
	}
	for i := range g {
		if g[i] != t[i] {
			return false
		}
	}
	return true
}

func (g Perm) String() string {
	return fmt.Sprint([]int(g))
}
