package drift

import "sync"

type Result struct {
	Index int
	Drift bool
	Stat  float64
	Err   error
}

func (d *Detector) CheckAll(windows [][]float64) []Result {

	results := make(chan Result)
	var wg sync.WaitGroup

	out := make([]Result, len(windows))

	for i, w := range windows {
		wg.Add(1)
		go func() {
			defer wg.Done()
			drift, stat, err := d.Check(w)
			r := Result{Index: i, Drift: drift, Stat: stat, Err: err}
			results <- r
		}()

	}

	go func() {
		wg.Wait()
		close(results)
	}()

	for res := range results {
		out[res.Index] = res
	}
	return out
}
