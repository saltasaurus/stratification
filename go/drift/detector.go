package drift

import (
	"errors"
	"slices"
)

var ErrThreshold = errors.New("drift: threshold must be >= 0.0")
var ErrNilStatistic = errors.New("drift: statistic cannot be nil")

type Detector struct {
	Reference []float64
	Threshold float64
	Stat      Statistic
}

func NewDetector(stat Statistic, ref []float64, threshold float64) (*Detector, error) {
	if stat == nil {
		return nil, ErrNilStatistic
	}

	if !(threshold >= 0.0) {
		return nil, ErrThreshold
	}

	if len(ref) == 0 {
		return nil, ErrEmpty
	}

	return &Detector{Stat: stat, Reference: slices.Clone(ref), Threshold: threshold}, nil
}

func (d *Detector) Check(window []float64) (drift bool, stat float64, err error) {
	stat, err = d.Stat.Compute(d.Reference, window)
	if err != nil {
		return false, 0.0, err
	}
	return stat > d.Threshold, stat, nil
}
