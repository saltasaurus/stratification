package drift

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

func ReadValues(r io.Reader) ([]float64, error) {
	sc := bufio.NewScanner(r)
	var vals []float64

	var lineNumber int

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		lineNumber++
		if line == "" {
			continue
		}
		val, err := strconv.ParseFloat(line, 64)
		if err != nil {
			return nil, fmt.Errorf("on line %d: %w", lineNumber, err)
		}

		vals = append(vals, val)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}

	if len(vals) == 0 {
		return nil, ErrEmpty
	}

	return vals, nil
}
