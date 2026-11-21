package hes

// Multi-radix heap MRH(r; C), the queue from the multi-radix heap note.
// This is the same structure as hodgemix-go/mrh.go. The mixer is package main,
// so the runner carries its own copy of the operations rather than importing it.
//
// Buckets form a k-level hierarchy with place values pi_0 = 1, pi_{j+1} = pi_j * r_j.
// Level j has r_j buckets of width pi_j covering [L_j, L_j+pi_{j+1}).
// The block of level j-1 is the active bucket of level j.
// An item sits in the least level whose block holds its key.
// Buckets are FIFO, so equal keys leave in insertion order.
// Keys inserted or decreased lie in [mu, mu+C].

import (
	"errors"
	"fmt"
)

type mrh struct {
	r     []int
	pi    []int
	c     int
	mu    int
	L     []int
	alpha []int
	key   []int
	lvl   []int
	bkt   []int
	next  []int
	prev  []int
	head  [][]int
	tail  [][]int
	n     int
}

// RadicesFor is the default radix tuple whose product covers the key universe
// [0, C]. Each radix is 2, so the product is the next power of two above C.
// A caller can pass its own tuple; the heap only requires each radix >= 2 and
// the product >= C+1.
func RadicesFor(c int) []int {
	if c < 0 {
		c = 0
	}
	r := []int{}
	p := 1
	for p < c+1 {
		r = append(r, 2)
		p *= 2
	}
	if len(r) == 0 {
		r = []int{2}
	}
	return r
}

func newMRH(r []int, c int) (*mrh, error) {
	if len(r) == 0 {
		return nil, errors.New("mrh: no radices")
	}
	if c < 0 {
		return nil, fmt.Errorf("mrh: negative window %d", c)
	}
	h := &mrh{r: append([]int(nil), r...), c: c, pi: make([]int, len(r)+1)}
	h.pi[0] = 1
	for j, rj := range r {
		if rj < 2 {
			return nil, fmt.Errorf("mrh: radix r_%d = %d < 2", j, rj)
		}
		h.pi[j+1] = h.pi[j] * rj
	}
	if h.pi[len(r)] < c+1 {
		return nil, fmt.Errorf("mrh: prod r = %d < C+1 = %d", h.pi[len(r)], c+1)
	}
	k := len(r)
	h.L, h.alpha = make([]int, k), make([]int, k)
	h.head, h.tail = make([][]int, k), make([][]int, k)
	for j := range r {
		h.head[j] = fillNeg(r[j])
		h.tail[j] = fillNeg(r[j])
	}
	return h, nil
}

func fillNeg(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = -1
	}
	return s
}

func (h *mrh) find(kappa int) (int, int, bool) {
	for j := range h.r {
		if kappa >= h.L[j] && kappa < h.L[j]+h.pi[j+1] {
			return j, (kappa - h.L[j]) / h.pi[j], true
		}
	}
	return -1, -1, false
}

func (h *mrh) grow(x int) {
	for len(h.key) <= x {
		h.key = append(h.key, 0)
		h.lvl = append(h.lvl, -1)
		h.bkt = append(h.bkt, -1)
		h.next = append(h.next, -1)
		h.prev = append(h.prev, -1)
	}
}

func (h *mrh) link(x, j, t int) {
	h.lvl[x], h.bkt[x], h.next[x], h.prev[x] = j, t, -1, h.tail[j][t]
	if h.tail[j][t] >= 0 {
		h.next[h.tail[j][t]] = x
	} else {
		h.head[j][t] = x
	}
	h.tail[j][t] = x
}

func (h *mrh) unlink(x int) {
	j, t := h.lvl[x], h.bkt[x]
	if h.prev[x] >= 0 {
		h.next[h.prev[x]] = h.next[x]
	} else {
		h.head[j][t] = h.next[x]
	}
	if h.next[x] >= 0 {
		h.prev[h.next[x]] = h.prev[x]
	} else {
		h.tail[j][t] = h.prev[x]
	}
	h.lvl[x], h.bkt[x], h.next[x], h.prev[x] = -1, -1, -1, -1
}

func (h *mrh) rebase() {
	var items []int
	for j := range h.r {
		for t := range h.r[j] {
			for x := h.head[j][t]; x >= 0; x = h.next[x] {
				items = append(items, x)
			}
		}
	}
	for _, x := range items {
		h.unlink(x)
	}
	for j := range h.r {
		h.L[j], h.alpha[j] = h.mu, 0
	}
	for _, x := range items {
		j, t, _ := h.find(h.key[x])
		h.link(x, j, t)
	}
}

func (h *mrh) insert(x, kappa int) error {
	if kappa < h.mu || kappa > h.mu+h.c {
		return fmt.Errorf("mrh: key %d outside [%d, %d]", kappa, h.mu, h.mu+h.c)
	}
	h.grow(x)
	if h.lvl[x] >= 0 {
		return fmt.Errorf("mrh: item %d already in the heap", x)
	}
	h.key[x] = kappa
	j, t, ok := h.find(kappa)
	if !ok {
		h.rebase()
		j, t, _ = h.find(kappa)
	}
	h.link(x, j, t)
	h.n++
	return nil
}

func (h *mrh) decrease(x, kappa int) error {
	if x >= len(h.lvl) || h.lvl[x] < 0 {
		return fmt.Errorf("mrh: item %d not in the heap", x)
	}
	if kappa < h.mu || kappa >= h.key[x] {
		return fmt.Errorf("mrh: decrease-key %d not in [%d, %d)", kappa, h.mu, h.key[x])
	}
	h.unlink(x)
	h.key[x] = kappa
	j, t, _ := h.find(kappa)
	h.link(x, j, t)
	return nil
}

func (h *mrh) extract() (x, kappa int, ok bool) {
	if h.n == 0 {
		return -1, 0, false
	}
	for {
		for t := h.alpha[0]; t < h.r[0]; t++ {
			if h.head[0][t] >= 0 {
				h.alpha[0] = t
				x = h.head[0][t]
				h.unlink(x)
				h.n--
				h.mu = h.key[x]
				return x, h.mu, true
			}
		}
		h.expand()
	}
}

func (h *mrh) expand() {
	for js := 1; js < len(h.r); js++ {
		for t := h.alpha[js] + 1; t < h.r[js]; t++ {
			if h.head[js][t] < 0 {
				continue
			}
			h.alpha[js] = t
			base := h.L[js] + t*h.pi[js]
			for i := js - 1; i >= 0; i-- {
				h.L[i], h.alpha[i] = base, 0
			}
			var items []int
			for y := h.head[js][t]; y >= 0; y = h.next[y] {
				items = append(items, y)
			}
			for _, y := range items {
				h.unlink(y)
				j, b, _ := h.find(h.key[y])
				h.link(y, j, b)
			}
			return
		}
	}
	panic("mrh: expand found no item although the heap is not empty")
}

// popOrder inserts keys[i] as item i and extracts them all.
func popOrder(keys, r []int, c int) ([]int, error) {
	h, err := newMRH(r, c)
	if err != nil {
		return nil, err
	}
	for i, k := range keys {
		if err := h.insert(i, k); err != nil {
			return nil, err
		}
	}
	out := make([]int, 0, len(keys))
	for h.n > 0 {
		x, _, _ := h.extract()
		out = append(out, x)
	}
	return out, nil
}
