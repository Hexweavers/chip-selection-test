from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Literal

from db import Repository, init_db

router = APIRouter()


def get_repo():
    db = init_db()
    return Repository(db)


@router.get("")
def get_stats(
    group_by: Literal["model", "persona_id", "style", "input_type"] = Query("model"),
    run_id: Optional[str] = Query(None),
):
    """Get aggregated stats for dashboard/leaderboard."""
    repo = get_repo()
    try:
        stats = repo.get_stats(group_by=group_by, run_id=run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"stats": stats}
