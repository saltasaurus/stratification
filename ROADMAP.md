# Stratification Roadmap

> Tentative. Package names are geology terms matched to their job; modules inside use plain ML names.

## Foundation
| Package | Job | Why the name |
|---|---|---|
| **bedrock** | Tensor + autograd library (Tensaur) | The foundation everything rests on |
| **strata** | DS fundamentals: `splits`, `sampling`, `preprocess`, `metrics`, `calibration`, `stats`, `quality` | Strata are layers laid down in order over time, like data; "stratified sampling" borrows the term |

## Data pipeline
| Package | Job | Why the name |
|---|---|---|
| **erosion** | Scraping + ingestion | Erosion carries material away from its source |
| **basin** | Raw data lake / warehouse | Where sediment collects before it's processed |
| **cement** | Feature pipelines + feature store | Cementation binds loose grains into solid rock |

## Models and evaluation
| Package | Job | Why the name |
|---|---|---|
| **mineral** | Models + training | Minerals form as atoms settle into ordered structure, like weights during training |
| **assay** | Evaluation harness + reports | An assay tests ore to grade its quality |
| **fossil** | Model registry + lineage | The fossil record is an ordered history of what existed when |

## Serving and operations
| Package | Job | Why the name |
|---|---|---|
| **convection** | Serving gateway / inference API | Moves material from depth to the surface |
| **vein** | Streaming / message bus | Veins are fractures that fluids flow through |
| **aquifer** | Cache for hot features + embeddings | A store you can draw from quickly |
| **uplift** | Canary / shadow rollout | Gradually raises a new model into view |
| **fault** | Retries, timeouts, circuit breakers | Faults are where rock breaks under stress (fault tolerance) |
| **seismic** | Observability: metrics, traces, alerts | Seismographs sense tremors before damage shows |
| **drift** | Data + concept drift monitors | Continents drift slowly; so do data distributions |

## AI systems
| Package | Job | Why the name |
|---|---|---|
| **orogeny** | Agents + LLM systems | Mountain building: everything below pushed together into something big |

## Dependency rule
`bedrock` ← `strata` ← `mineral` / `assay` ← `orogeny`

Pipeline and ops packages sit alongside and may import downward only.

## Current priority
Real now: `bedrock`, `strata`, `mineral`, `convection`, `drift`. Everything else stays roadmap until a project needs it.
