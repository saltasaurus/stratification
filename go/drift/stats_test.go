package drift

import (
	"errors"
	"math"
	"slices"
	"testing"
)

const tol = 1e-12

func checkNoError(t *testing.T, err error, f string) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: unexpected error: %v", f, err)
	}
}

func TestMean(t *testing.T) {
	got, err := Mean([]float64{1, 2, 3, 4})
	checkNoError(t, err, "Mean")
	want := 2.5
	if got != want {
		t.Errorf("Mean = %v, want %v", got, want)
	}
}

func TestVariance(t *testing.T) {
	got, err := Variance([]float64{0.1, 0.2, 0.3})
	checkNoError(t, err, "Variance")
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
	got, err := EmpiricalCDF(x, []float64{0, 1, 2, 2.5, 3, 4})
	checkNoError(t, err, "EmpiricalCDF")
	want := []float64{0, 0.25, 0.75, 0.75, 1, 1}
	if !slices.Equal(got, want) {
		t.Errorf("EmpiricalCDF = %v, want %v", got, want)
	}
	if !slices.Equal(x, []float64{2, 1, 3, 2}) {
		t.Errorf("input was modified: %v", x)
	}
}

func TestKSStatistic(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	b := []float64{3, 4, 5, 6}
	cases := []struct {
		name string
		x, y []float64
		want float64
	}{
		{"half shift", a, b, 0.5},
		{"symmetric", b, a, 0.5},
		{"identical", a, a, 0},
		{"no overlap", []float64{1, 2}, []float64{10, 20}, 1},
	}
	for _, c := range cases {
		got, err := KSStatistic(c.x, c.y)
		checkNoError(t, err, c.name)

		if math.Abs(got-c.want) > tol {
			t.Errorf("%s: KSStatistic = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestEmptyInput(t *testing.T) {
	if _, err := Mean(nil); !errors.Is(err, ErrEmpty) {
		t.Errorf("Mean(nil) err = %v, want ErrEmpty", err)
	}
	if _, err := KSStatistic(nil, []float64{1, 2}); !errors.Is(err, ErrEmpty) {
		t.Errorf("KSStatistic(nil, ...) err = %v, want ErrEmpty", err)
	}
	if _, err := KSStatistic([]float64{1, 2}, []float64{}); !errors.Is(err, ErrEmpty) {
		t.Errorf("KSStatistic(..., empty) err = %v, want ErrEmpty", err)
	}
}
