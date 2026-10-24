package drift

import (
	"math"
	"slices"
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

func TestSortedCopy(t *testing.T) {
	x := []float64{3, 1, 2}
	got := sortedCopy(x)
	if !slices.Equal(got, []float64{1, 2, 3}) {
		t.Errorf("sortedCopy = %v, want [1 2 3]", got)
	}
	if !slices.Equal(x, []float64{3, 1, 2}) {
		t.Errorf("input was modified: %v", x)
	}
}

func TestEmpiricalCDF(t *testing.T) {
	x := []float64{2, 1, 3, 2}
	got := EmpiricalCDF(x, []float64{0, 1, 2, 2.5, 3, 4})
	want := []float64{0, 0.25, 0.75, 0.75, 1, 1}
	if !slices.Equal(got, want) {
		t.Errorf("EmpiricalCDF = %v, want %v", got, want)
	}
	if !slices.Equal(x, []float64{2, 1, 3, 2}) {
		t.Errorf("input was modified: %v", x)
	}
}
