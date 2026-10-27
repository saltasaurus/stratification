package api

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const tol = 1e-9

// post sends body to HandleDrift and returns the recorded response.
func post(t *testing.T, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/drift", strings.NewReader(body))
	rec := httptest.NewRecorder()
	HandleDrift(rec, req)
	return rec
}

func TestHandleDriftOK(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		wantStat  float64
		wantDrift bool
	}{
		{"ks", `{"ref":[1,2,3,4],"window":[3,4,5,6],"stat":"ks","threshold":0.1}`, 0.5, true},
		{"mean", `{"ref":[1,2,3,4],"window":[3,4,5,6],"stat":"mean","threshold":1.5}`, 2, true},
		{"default stat is ks", `{"ref":[1,2,3,4],"window":[3,4,5,6]}`, 0.5, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := post(t, tt.body)
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body: %s", rec.Code, rec.Body.String())
			}
			if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
				t.Errorf("Content-Type = %q, want application/json", ct)
			}
			var got driftResponse
			if err := json.NewDecoder(rec.Body).Decode(&got); err != nil {
				t.Fatalf("decoding response: %v", err)
			}
			if math.Abs(got.Stat-tt.wantStat) > tol || got.Drift != tt.wantDrift {
				t.Errorf("got %+v, want stat=%v drift=%v", got, tt.wantStat, tt.wantDrift)
			}
		})
	}
}

func TestHandleDriftBadRequest(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"broken JSON", `{"ref":[1,2`},
		{"empty window", `{"ref":[1,2,3],"window":[]}`},
		{"unknown stat", `{"ref":[1,2,3],"window":[4,5],"stat":"foo"}`},
		{"negative threshold", `{"ref":[1,2,3],"window":[4,5],"threshold":-1}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := post(t, tt.body)
			if rec.Code != http.StatusBadRequest {
				t.Errorf("status = %d, want 400; body: %s", rec.Code, rec.Body.String())
			}
		})
	}
}
