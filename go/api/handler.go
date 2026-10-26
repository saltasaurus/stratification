package api

import (
	"encoding/json"
	"net/http"
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

func HandleDrift(w http.ResponseWriter, r *http.Request) {
	request := driftRequest{Stat: "ks", Threshold: 0.1}
	err := json.NewDecoder(r.Body).Decode(&request)

	if err != nil {
		http.Error(w, "decoder error", http.StatusBadRequest)
		return
	}

}
