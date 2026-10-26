package drift

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

func randomSample(r *rand.Rand, n int, shift float64) []float64 {
	x := make([]float64, n)
	for i := range len(x) {
		x[i] = r.NormFloat64() + shift
	}
	return x
}

func BenchmarkKSStatistic(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2)) // fixed seed: same data every run
	for _, n := range []int{100, 1_000, 10_000} {
		ref := randomSample(r, n, 0)
		win := randomSample(r, n, 0.5)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for b.Loop() { // Go 1.24+: setup above isn't timed
				if _, err := KSStatistic(ref, win); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
