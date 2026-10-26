package drift

import (
	"errors"
	"math"
	"testing"
)

var _ Statistic = KS{} // compile-time check: fails to build if KS stops satisfying Statistic
var _ Statistic = MeanShift{}

func TestStatistics(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	b := []float64{3, 4, 5, 6}
	cases := []struct {
		name string
		stat Statistic
		want float64
	}{
		{"ks", KS{}, 0.5},
		{"mean shift", MeanShift{}, 2},
	}
	for _, c := range cases {
		got, err := c.stat.Compute(a, b)
		checkNoError(t, err, c.name)
		if math.Abs(got-c.want) > tol {
			t.Errorf("%s: Compute = %v, want %v", c.name, got, c.want)
		}
		if _, err := c.stat.Compute(a, nil); !errors.Is(err, ErrEmpty) {
			t.Errorf("%s: empty window err = %v, want ErrEmpty", c.name, err)
		}
	}
}

func TestDetectorMeanShift(t *testing.T) {
	d, err := NewDetector(MeanShift{}, []float64{1, 2, 3, 4}, 1.0)
	checkNoError(t, err, "NewDetector")
	drift, stat, err := d.Check([]float64{3, 4, 5, 6})
	checkNoError(t, err, "Check")
	if !drift || math.Abs(stat-2) > tol {
		t.Errorf("Check = (%v, %v), want (true, 2)", drift, stat)
	}
}
