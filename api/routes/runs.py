from fastapi import APIRouter, HTTPException, Query

from db import Repository, init_db

router = APIRouter()


def get_repo():
    db = init_db()
    return Repository(db)


@router.get("")
def list_runs(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all test runs."""
    repo = get_repo()
    runs = repo.list_runs(limit=limit, offset=offset)
    return {"runs": runs}


@router.get("/{run_id}")
def get_run(run_id: str):
    """Get a single run by ID."""
    repo = get_repo()
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "id": run.id,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "status": run.status,
        "config": run.config,
    }
