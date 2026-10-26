package drift

import (
	"errors"
	"math"
	"testing"
)

func TestNewDetector(t *testing.T) {
	if _, err := NewDetector(KS{}, nil, 0.3); !errors.Is(err, ErrEmpty) {
		t.Errorf("empty ref: err = %v, want ErrEmpty", err)
	}
	if _, err := NewDetector(KS{}, []float64{1, 2}, -0.1); !errors.Is(err, ErrThreshold) {
		t.Errorf("threshold -0.1: err = %v, want ErrThreshold", err)
	}
	if _, err := NewDetector(nil, []float64{1, 2}, 0.5); !errors.Is(err, ErrNilStatistic) {
		t.Errorf("nil stat: err = %v, want ErrNilStatistic", err)
	}
	ref := []float64{1, 2, 3, 4}
	d, err := NewDetector(KS{}, ref, 0.3)
	checkNoError(t, err, "NewDetector")
	ref[0] = 100
	if d.Reference[0] != 1 {
		t.Errorf("detector shares caller's slice: Reference[0] = %v", d.Reference[0])
	}
}

func TestDetectorCheck(t *testing.T) {
	d, err := NewDetector(KS{}, []float64{1, 2, 3, 4}, 0.3)
	checkNoError(t, err, "NewDetector")
	cases := []struct {
		name      string
		window    []float64
		wantDrift bool
		wantStat  float64
	}{
		{"shifted", []float64{3, 4, 5, 6}, true, 0.5},
		{"same", []float64{1, 2, 3, 4}, false, 0},
	}
	for _, c := range cases {
		drift, stat, err := d.Check(c.window)
		checkNoError(t, err, c.name)
		if drift != c.wantDrift || math.Abs(stat-c.wantStat) > tol {
			t.Errorf("%s: Check = (%v, %v), want (%v, %v)", c.name, drift, stat, c.wantDrift, c.wantStat)
		}
	}
	if _, _, err := d.Check(nil); !errors.Is(err, ErrEmpty) {
		t.Errorf("empty window: err = %v, want ErrEmpty", err)
	}
}
