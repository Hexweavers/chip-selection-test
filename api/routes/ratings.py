from fastapi import APIRouter, Query
from typing import Optional

from db import Repository, init_db

router = APIRouter()


def get_repo():
    db = init_db()
    return Repository(db)


@router.get("")
def list_ratings(
    result_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List ratings with filters."""
    repo = get_repo()
    ratings = repo.get_ratings(
        result_id=result_id,
        user_id=user_id,
        limit=limit,
        offset=offset,
    )
    return {"ratings": ratings}
