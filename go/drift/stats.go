package drift

import (
	"errors"
	"math"
	"slices"
	"sort"
)

var ErrEmpty = errors.New("drift: empty sample")

func Mean(x []float64) (float64, error) {
	n := len(x)

	if n == 0 {
		return 0.0, ErrEmpty
	}

	var sum float64
	for _, val := range x {
		sum += val
	}
	return sum / float64(n), nil
}

func Variance(x []float64) (float64, error) {
	mean, err := Mean(x)

	if err != nil {
		return 0.0, err
	}

	var xs = make([]float64, 0, len(x))
	for _, val := range x {
		d := val - mean
		xs = append(xs, d*d)
	}

	return Mean(xs)
}

func sortedCopy(x []float64) []float64 {

	dst := slices.Clone(x)
	slices.Sort(dst)

	return dst
}

func EmpiricalCDF(x []float64, t []float64) ([]float64, error) {
	n := len(x)
	m := len(t)

	if n == 0 {
		return nil, ErrEmpty
	}

	dst := make([]float64, m)

	sortedX := sortedCopy(x)

	for j, val := range t {
		count := sort.Search(len(sortedX), func(i int) bool { return sortedX[i] > val })
		dst[j] = float64(count) / float64(n)
	}
	return dst, nil
}

func KSStatistic(x, y []float64) (float64, error) {

	if len(x) == 0 || len(y) == 0 {
		return 0.0, ErrEmpty
	}

	xy := slices.Concat(x, y)

	fn, err := EmpiricalCDF(x, xy)
	if err != nil {
		return 0.0, err
	}
	gm, err := EmpiricalCDF(y, xy)
	if err != nil {
		return 0.0, err
	}

	var sup float64

	for i := range xy {
		sup = max(sup, math.Abs(fn[i]-gm[i]))
	}
	return sup, nil
}
