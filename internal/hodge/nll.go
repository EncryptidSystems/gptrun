package hodge

import (
	"fmt"
	"math"
)

// textNLL is the teacher-forced negative log likelihood of one text, in nats,
// and the number of predicted tokens. The head is the full softmax.
func (m *Model) textNLL(text string) (float64, int, error) {
	ids := append(append([]int{}, m.bos...), m.tok.Encode(text, true)...)
	if len(ids) < 2 {
		return 0, 0, nil
	}
	d := &dialog{}
	req := &headReq{exact: true}
	var sum float64
	for i := 0; i < len(ids)-1; i++ {
		if err := m.feed(d, ids[i], req); err != nil {
			return 0, 0, err
		}
		sum -= logsoftAt(d.lg, ids[i+1])
	}
	return sum, len(ids) - 1, nil
}

// ContNLL is the teacher-forced negative log likelihood of cont given ctx,
// in nats, and the number of continuation tokens. The head is the full softmax.
func ContNLL(name, ctx, cont string) (float64, int, error) {
	liveMu.Lock()
	defer liveMu.Unlock()
	var sum float64
	var n int
	err := do(name, func(m *Model) error {
		var e error
		sum, n, e = m.contNLL(ctx, cont)
		return e
	}, func() bool { return true })
	return sum, n, err
}

// contNLL scores cont after ctx. A cached prefix is kept through the token
// before the last context token. That last context token is run again so the
// first continuation token is scored from its hidden state, not from a
// previous choice that continued past it.
func (m *Model) contNLL(ctx, cont string) (float64, int, error) {
	kids := m.tok.Encode(cont, true)
	if len(kids) == 0 {
		return 0, 0, fmt.Errorf("empty continuation")
	}
	ids := append(append([]int{}, m.bos...), m.tok.Encode(ctx, true)...)
	ctxN := len(ids)
	if ctxN < 1 {
		return 0, 0, fmt.Errorf("empty context")
	}
	ids = append(ids, kids...)
	d := &dialog{}
	keep := ctxN - 1
	if keep > 0 && m.gpu != nil {
		c := commonPrefix(m.gpu.kv, ids[:keep])
		d.ids = append([]int(nil), ids[:c]...)
		d.pos = c
	}
	req := &headReq{exact: true}
	if err := m.step(d, ids[d.pos:ctxN], req); err != nil {
		return 0, 0, err
	}
	sum := -logsoftAt(d.lg, ids[ctxN])
	for i := ctxN; i < len(ids)-1; i++ {
		if err := m.feed(d, ids[i], req); err != nil {
			return 0, 0, err
		}
		sum -= logsoftAt(d.lg, ids[i+1])
	}
	return sum, len(kids), nil
}

func logsoftAt(lg []float32, id int) float64 {
	if id < 0 || id >= len(lg) || len(lg) == 0 {
		return math.Inf(-1)
	}
	max := math.Inf(-1)
	for _, v := range lg {
		if x := float64(v); x > max {
			max = x
		}
	}
	var z float64
	for _, v := range lg {
		z += math.Exp(float64(v) - max)
	}
	return float64(lg[id]) - max - math.Log(z)
}
