from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import runs, results, ratings, stats, options, generate
from db import get_db

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
app.include_router(options.router, prefix="/options", tags=["options"])
app.include_router(generate.router, prefix="/generate", tags=["generate"])


@app.get("/health")
def health():
    try:
        db = get_db()
        db.execute("SELECT 1").fetchone()
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "database": "disconnected", "error": str(e)},
        )
