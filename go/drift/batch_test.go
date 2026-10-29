package drift

import (
	"errors"
	"testing"
)

func newKS(t *testing.T) *Detector {
	t.Helper()
	d, err := NewDetector(KS{}, []float64{1, 2, 3, 4}, 0.1)
	if err != nil {
		t.Fatalf("NewDetector: %v", err)
	}
	return d
}

// CheckAll must give exactly what calling Check one window at a time gives.
func TestCheckAllMatchesCheck(t *testing.T) {
	d := newKS(t)
	windows := [][]float64{
		{3, 4, 5, 6},
		{1, 2, 3, 4},
		{},
		{10, 11, 12},
	}

	got := d.CheckAll(windows)
	if len(got) != len(windows) {
		t.Fatalf("len = %d, want %d", len(got), len(windows))
	}

	for i, w := range windows {
		wantDrift, wantStat, wantErr := d.Check(w)
		r := got[i]
		if r.Index != i {
			t.Errorf("result %d: Index = %d", i, r.Index)
		}
		if r.Drift != wantDrift || r.Stat != wantStat {
			t.Errorf("result %d: got (%v, %v), want (%v, %v)", i, r.Drift, r.Stat, wantDrift, wantStat)
		}
		if !errors.Is(r.Err, wantErr) {
			t.Errorf("result %d: Err = %v, want %v", i, r.Err, wantErr)
		}
	}

	if !errors.Is(got[2].Err, ErrEmpty) {
		t.Errorf("empty window: Err = %v, want ErrEmpty", got[2].Err)
	}
}

func TestCheckAllEmptyInput(t *testing.T) {
	d := newKS(t)
	for _, in := range [][][]float64{nil, {}} {
		if got := d.CheckAll(in); len(got) != 0 {
			t.Errorf("CheckAll(%v) len = %d, want 0", in, len(got))
		}
	}
}

// Many goroutines finishing in random order must still land in input order.
func TestCheckAllManyKeepsOrder(t *testing.T) {
	d := newKS(t)
	windows := make([][]float64, 1000)
	for i := range windows {
		shift := float64(i % 5)
		windows[i] = []float64{1 + shift, 2 + shift, 3 + shift, 4 + shift}
	}

	got := d.CheckAll(windows)
	for i, r := range got {
		if r.Index != i {
			t.Fatalf("got[%d].Index = %d", i, r.Index)
		}
		_, wantStat, _ := d.Check(windows[i])
		if r.Stat != wantStat {
			t.Fatalf("got[%d].Stat = %v, want %v", i, r.Stat, wantStat)
		}
	}
}
