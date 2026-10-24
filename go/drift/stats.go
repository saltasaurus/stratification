package drift

import (
	"math"
	"slices"
	"sort"
)

func Mean(x []float64) float64 {
	n := len(x)

	if n == 0 {
		return 0.0
	}

	var sum float64
	for _, val := range x {
		sum += val
	}
	return sum / float64(n)
}

func Variance(x []float64) float64 {
	mean := Mean(x)

	var xs []float64
	for _, val := range x {
		xs = append(xs, math.Pow(val-mean, 2))
	}

	return Mean(xs)
}

func sortedCopy(x []float64) []float64 {

	dst := slices.Clone(x)
	slices.Sort(dst)

	return dst
}

func EmpiricalCDF(x []float64, t []float64) float64 {
	i := sort.Search(len(x), func(i int) bool { return x[i] > t })
	return float64(i) / float64(len(x))
}
