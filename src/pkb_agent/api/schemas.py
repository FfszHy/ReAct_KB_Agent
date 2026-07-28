"""Request contracts for the browser-facing API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StartRunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=12_000)
    user_id: str = Field(default="default", min_length=1, max_length=200)


class ApprovalDecisionRequest(BaseModel):
    approved: bool
