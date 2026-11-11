This roadmap is tentative and far into the horizon. Names, ideas, and goals may change at any time

## Fundamentals
- Sediment: Web scraper + raw ingestion ETL; landing-zone data warehouse / lake of unprocessed source material
- Matrix: Feature store / embedding store; the base tabular + vector substrate that holds derived features and labels
- Facies: Environment-specific feature sets (e.g. time-series facies, text facies, image facies); domain-adapted transformers or preprocessing suites
- Core: Continuous vertical data sampler / windowing library; core-sample style train/val/test splits that preserve stratigraphic order
- Lithology: Schema registry + type system; material classification & validation of incoming data shapes

## Infrastructure & serving
- Outcrop: Model serving gateway / inference API surface; the public endpoint that exposes trained models
- Vein: High-throughput message bus / streaming backbone (Kafka-style or NATS); focused data-movement pathways
- Aquifer: Distributed cache + feature retrieval service; high-permeability store for hot features and embeddings
- Exposure: Canary / shadow traffic router; controlled rollout and A/B exposure of model versions
- Host: Container orchestration + runtime environment (the surrounding system that hosts model pods)
- Conduit: gRPC/HTTP client libraries + service mesh connectors; typed inter-service pathways

## Specialized
- Drift: Concept/data-drift detector; statistical monitors that flag distributional shifts in inputs or predictions
- Inclusion: Anomaly detection service; isolation of outliers, rare events, and out-of-distribution samples
- Seismic: Observability / metrics + tracing stack; continuous sensing of latency, error rates, and internal model health
- Assay: Evaluation harness; offline metric computation, backtesting, and grade (quality) reporting
- Succession: Model registry + version lineage; ordered stacking of successive model artifacts and their metadata
- Diagenesis: Feature transformation / featurization pipelines; post-ingestion alteration, enrichment, and materialization jobs
- Basin: Long-term archival store / data lakehouse; large-scale accumulation of historical features, predictions, and logs
- Crystallization: Training orchestration + hyperparameter optimization runner; ordered growth of model parameters under constrained compute
