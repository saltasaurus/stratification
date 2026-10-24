package drift

import (
	"math"
	"testing"
)

const tol = 1e-12

func TestMean(t *testing.T) {
	got := Mean([]float64{1, 2, 3, 4})
	want := 2.5
	if got != want {
		t.Errorf("Mean = %v, want %v", got, want)
	}
}

func TestVariance(t *testing.T) {
	got := Variance([]float64{0.1, 0.2, 0.3})
	want := 0.02 / 3
	if math.Abs(got-want) > tol {
		t.Errorf("Variance = %v, want %v", got, want)
	}
}
