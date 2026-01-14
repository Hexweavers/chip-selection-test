from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import runs, results, ratings, stats

app = FastAPI(
    title="Chip Benchmark API",
    description="Backend for chip generation benchmark results and ratings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Internal VPC only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router, prefix="/runs", tags=["runs"])
app.include_router(results.router, prefix="/results", tags=["results"])
app.include_router(ratings.router, prefix="/ratings", tags=["ratings"])
app.include_router(stats.router, prefix="/stats", tags=["stats"])


@app.get("/health")
def health():
    return {"status": "ok"}
