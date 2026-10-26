package drift

import "math"

type Statistic interface {
	Compute(ref, window []float64) (float64, error)
}

type KS struct{}

func (KS) Compute(ref, window []float64) (float64, error) {
	return KSStatistic(ref, window)
}

type MeanShift struct{}

func (MeanShift) Compute(ref, window []float64) (float64, error) {
	meanref, err := Mean(ref)
	if err != nil {
		return 0.0, err
	}

	meanwin, err := Mean(window)
	if err != nil {
		return 0.0, err
	}

	return math.Abs(meanref - meanwin), nil
}
