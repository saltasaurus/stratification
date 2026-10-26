package drift

import (
	"errors"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func TestReadValues(t *testing.T) {
	got, err := ReadValues(strings.NewReader("1\n2.5\n\n  3  \n"))
	checkNoError(t, err, "ReadValues")
	if !slices.Equal(got, []float64{1, 2.5, 3}) {
		t.Errorf("ReadValues = %v, want [1 2.5 3]", got)
	}

	if _, err := ReadValues(strings.NewReader("\n\n")); !errors.Is(err, ErrEmpty) {
		t.Errorf("blank input: err = %v, want ErrEmpty", err)
	}

	_, err = ReadValues(strings.NewReader("1\nabc\n"))
	if !errors.Is(err, strconv.ErrSyntax) {
		t.Errorf("bad number: err = %v, want wrapped strconv.ErrSyntax", err)
	}
	if err == nil || !strings.Contains(err.Error(), "line 2") {
		t.Errorf("bad number: err = %v, want it to mention line 2", err)
	}
}
