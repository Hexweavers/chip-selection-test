from fastapi import APIRouter, HTTPException, Query, Header
from typing import Optional
from pydantic import BaseModel, Field

from db import Repository, init_db

router = APIRouter()


def get_repo():
    db = init_db()
    return Repository(db)


class RatingCreate(BaseModel):
    rating: int = Field(..., ge=1, le=5)


@router.get("")
def list_results(
    run_id: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    persona_id: Optional[str] = Query(None),
    style: Optional[str] = Query(None),
    input_type: Optional[str] = Query(None),
    constraint_type: Optional[str] = Query(None),
    chip_count: Optional[int] = Query(None),
    rated_by: Optional[str] = Query(None),
    unrated_by: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    x_user_id: Optional[str] = Header(None),
):
    """List results with filters."""
    repo = get_repo()
    results, total = repo.list_results(
        run_id=run_id,
        model=model,
        persona_id=persona_id,
        style=style,
        input_type=input_type,
        constraint_type=constraint_type,
        chip_count=chip_count,
        rated_by=rated_by,
        unrated_by=unrated_by,
        limit=limit,
        offset=offset,
        user_id=x_user_id,
    )
    return {"results": results, "total": total}


@router.get("/{result_id}")
def get_result(result_id: str):
    """Get a single result with full details."""
    repo = get_repo()
    result = repo.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


@router.post("/{result_id}/ratings")
def create_rating(
    result_id: str,
    body: RatingCreate,
    x_user_id: str = Header(...),
):
    """Submit a rating for a result."""
    repo = get_repo()

    # Check result exists
    result = repo.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    rating_id, created_at = repo.add_rating(result_id, x_user_id, body.rating)

    return {
        "id": rating_id,
        "result_id": result_id,
        "user_id": x_user_id,
        "rating": body.rating,
        "created_at": created_at,
    }
