package main

import (
	"log"
	"net/http"
	"stratification/api"
)

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /drift", api.HandleDrift)

	log.Fatal(http.ListenAndServe(":8080", mux))
}
