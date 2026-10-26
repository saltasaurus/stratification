package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"stratification/drift"
)

func run() error {
	ref := flag.String("ref", "", "The reference file path")
	window := flag.String("window", "", "The window file path")
	stat := flag.String("stat", "ks", "The Statistic to use [ks, mean]")
	threshold := flag.Float64("threshold", 0.1, "Threshold for drift calculation")

	flag.Parse()

	if *ref == "" {
		return errors.New("-ref is required")
	}
	if *window == "" {
		return errors.New("-window is required")
	}

	r, err := readFile(*ref)
	if err != nil {
		return err
	}
	w, err := readFile(*window)
	if err != nil {
		return err
	}
	var s drift.Statistic
	switch *stat {
	case "ks":
		s = drift.KS{}
	case "mean":
		s = drift.MeanShift{}
	default:
		return fmt.Errorf("unknown -stat %q", *stat)
	}

	d, err := drift.NewDetector(s, r, *threshold)
	if err != nil {
		return err
	}

	isDrift, result, err := d.Check(w)

	if err != nil {
		return err
	}

	fmt.Printf("stat=%.4f drift=%v\n", result, isDrift)

	return nil

}

func readFile(path string) ([]float64, error) {
	f, err := os.Open(path)

	if err != nil {
		return nil, err
	}

	defer f.Close()

	vals, err := drift.ReadValues(f)

	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", path, err)
	}

	return vals, nil

}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "drift:", err)
		os.Exit(1)
	}
}
