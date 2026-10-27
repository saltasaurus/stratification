package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"stratification/drift"
)

type driftRequest struct {
	Reference []float64 `json:"ref"`
	Window    []float64 `json:"window"`
	Stat      string    `json:"stat"`
	Threshold float64   `json:"threshold"`
}

type driftResponse struct {
	Stat  float64 `json:"stat"`
	Drift bool    `json:"drift"`
}

func statusFor(err error) int {
	if errors.Is(err, drift.ErrEmpty) || errors.Is(err, drift.ErrThreshold) {
		return http.StatusBadRequest
	}
	return http.StatusInternalServerError
}

func HandleDrift(w http.ResponseWriter, r *http.Request) {
	req := driftRequest{Stat: "ks", Threshold: 0.1}
	err := json.NewDecoder(r.Body).Decode(&req)

	if err != nil {
		http.Error(w, "api: "+err.Error(), http.StatusBadRequest)
		return
	}

	var s drift.Statistic
	switch req.Stat {
	case "ks":
		s = drift.KS{}
	case "mean":
		s = drift.MeanShift{}
	default:
		http.Error(w, fmt.Sprintf("unknown stat %q", req.Stat), http.StatusBadRequest)
		return
	}

	d, err := drift.NewDetector(s, req.Reference, req.Threshold)

	if err != nil {
		errorCode := statusFor(err)
		http.Error(w, "api: "+err.Error(), errorCode)
		return
	}

	isDrift, statVal, err := d.Check(req.Window)

	if err != nil {
		errorCode := statusFor(err)
		http.Error(w, "api: "+err.Error(), errorCode)
		return
	}

	response := driftResponse{Stat: statVal, Drift: isDrift}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
