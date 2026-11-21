package hodge

import "testing"

func TestChunkFor(t *testing.T) {
	if chunkFor(0) != maxBatch || chunkFor(30*4) != maxBatch {
		t.Fatalf("one-plane chart: %d %d", chunkFor(0), chunkFor(30*4))
	}
	// 36 layers, four fused linears, six planes. That is hodge-5.1.
	if got := chunkFor(36 * 4 * 6); got != 3 {
		t.Fatalf("hodge-5.1 submit = %d, want 3", got)
	}
	if chunkFor(1<<20) != 1 {
		t.Fatalf("heavy chart: %d", chunkFor(1<<20))
	}
}
