package hodge

import (
	"fmt"
	"os"

	"gptrun/internal/rt"
)

// The device has one KV cache. A side pass (the route pass, a recipe or exec
// call) writes over the chat's rows from the first token they differ, and the
// chat then had to run its whole history again: every turn re-read the
// conversation. The shelf keeps the rows a sequence is about to lose in a
// device buffer, and copies them back when that sequence returns. A copy is
// the rows' bits, so a dialog that gets its rows from the shelf sees what a
// replay would have computed.
//
// Two buffers take turns: a swap puts the cache's rows on the free one and
// copies the shelved rows back from the other in the same submit.

// shelfRows is the most rows one shelf holds. Rows past it are replayed.
const shelfRows = 512

type shelf struct {
	ids  []int // the cache's token list when shelved
	from int   // first shelved position
	n    int   // rows shelved: positions from..from+n
	buf  int   // which of the two buffers
	// hi is the highest position written since. The rows the shelf does not
	// hold, from+n up to len(ids), are still in the cache while hi <= from+n.
	hi int
}

type shelfRT struct {
	kc, vc uint32
	bk, bv [2]uint32
	row    int // bytes per position per layer
	per    int // bytes per layer in kc and vc
	layers int
	rows   int
	cur    *shelf
}

// bootShelf allocates the two buffers. Without them a dialog replays.
// MODELRUN_SHELF=0 turns it off.
func (g *gpuRT) bootShelf(m *Model) error {
	if os.Getenv("MODELRUN_SHELF") == "0" {
		return nil
	}
	s := &shelfRT{row: m.KV * m.HD * 4, layers: m.L, rows: min(shelfRows, g.ctx)}
	s.per = g.ctx * s.row
	var err error
	if s.kc, err = g.dev.Buf("kc", nil, 16); err != nil {
		return err
	}
	if s.vc, err = g.dev.Buf("vc", nil, 16); err != nil {
		return err
	}
	size := s.layers * s.rows * s.row
	for i := 0; i < 2; i++ {
		if s.bk[i], err = g.dev.Buf(fmt.Sprintf("shelfk%d", i), nil, size); err != nil {
			return err
		}
		if s.bv[i], err = g.dev.Buf(fmt.Sprintf("shelfv%d", i), nil, size); err != nil {
			return err
		}
	}
	g.shelf = s
	logf("kv shelf  %d rows  %d MB", s.rows, 4*size>>20)
	return nil
}

// noteWrite records rows [from, to) written to the cache.
func (s *shelfRT) noteWrite(to int) {
	if s != nil && s.cur != nil && to > s.cur.hi {
		s.cur.hi = to
	}
}

// copies moves n rows per layer from position at in one buffer pair to
// position to in another.
func (s *shelfRT) copies(ops []rt.Op, srcK, srcV uint32, srcPer, srcAt int, dstK, dstV uint32, dstPer, dstAt, n int) []rt.Op {
	bytes := uint32(n * s.row)
	for l := 0; l < s.layers; l++ {
		so := uint32(l*srcPer + srcAt*s.row)
		do := uint32(l*dstPer + dstAt*s.row)
		ops = append(ops,
			rt.Op{Tag: rt.OpCopy, Src: srcK, SrcOff: so, Dst: dstK, DstOff: do, N: bytes},
			rt.Op{Tag: rt.OpCopy, Src: srcV, SrcOff: so, Dst: dstV, DstOff: do, N: bytes})
	}
	return ops
}

// fit readies the cache for a dialog that wants want in it. The rows past the
// prefix it shares with want are shelved, and when the shelf holds more of want
// than the cache does, those rows come back. g.kv is then the cache's token
// list; the caller reuses its common prefix with want and runs the rest.
func (m *Model) fit(want []int) error {
	g := m.gpu
	s := g.shelf
	if s == nil {
		return nil
	}
	c := commonPrefix(g.kv, want)
	if c >= len(want) {
		return nil
	}
	// How far want gets with the shelf: its rows, and past them the rows
	// nobody wrote over.
	k := c
	old := s.cur
	if old != nil && old.from <= c {
		reach := len(old.ids)
		if old.hi > old.from+old.n {
			reach = old.from + old.n
		}
		k = max(c, min(commonPrefix(old.ids, want), reach))
	}
	// The cache's rows from c on, which want would write over.
	n := min(len(g.kv)-c, s.rows)
	if n <= 0 && k == c {
		return nil
	}
	var ops []rt.Op
	next := old
	if n > 0 {
		free := 0
		if old != nil {
			free = 1 - old.buf
		}
		ops = s.copies(ops, s.kc, s.vc, s.per, c, s.bk[free], s.bv[free], s.rows*s.row, 0, n)
		next = &shelf{ids: append([]int(nil), g.kv...), from: c, n: n, buf: free, hi: c + n}
	}
	if k > c {
		// The shelved rows [c, min(k, from+n)) go back. Past from+n they are
		// still in the cache.
		back := min(k, old.from+old.n) - c
		if back > 0 {
			ops = s.copies(ops, s.bk[old.buf], s.bv[old.buf], s.rows*s.row, c-old.from, s.kc, s.vc, s.per, c, back)
		}
		if next == old {
			// Nothing new was shelved: the old shelf is spent.
			next = nil
		} else {
			next.hi = max(next.hi, c+back)
		}
		logf("kv shelf  %d rows back", k-c)
		g.kv = append(g.kv[:0], want[:k]...)
	} else {
		g.kv = g.kv[:c]
	}
	if len(ops) > 0 {
		if _, err := g.dev.Run2(nil, ops, nil, false); err != nil {
			return err
		}
	}
	s.cur = next
	return nil
}
