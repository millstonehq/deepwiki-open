"""
FastAPI endpoints for server-side wiki generation.

Provides:
- POST /api/generate - Start wiki generation job
- GET /api/generate/{job_id}/status - Poll job status
- GET /api/generate/{job_id}/result - Get completed wiki
- GET /api/generate/{job_id}/progress - SSE progress stream
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.job_manager import job_manager, JobStatus
from api.wiki_generator import WikiGenerator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["generate"])


class GenerateRequest(BaseModel):
    """Request model for wiki generation."""
    repo_url: str = Field(..., description="Repository URL (GitHub, GitLab, or Bitbucket)")
    branch: str = Field("main", description="Branch to generate wiki from")
    token: Optional[str] = Field(None, description="Access token for private repositories")
    format: str = Field("markdown", description="Output format: 'markdown' or 'json'")
    language: str = Field("en", description="Language code for generated content")
    provider: str = Field("vertex", description="LLM provider to use")
    model: Optional[str] = Field(None, description="Model name (uses default if not specified)")
    comprehensive: bool = Field(True, description="Generate comprehensive (8-12 pages) or concise (4-6 pages) wiki")
    excluded_dirs: Optional[str] = Field(None, description="Directories to exclude (newline-separated)")
    included_files: Optional[str] = Field(None, description="Files to include exclusively (newline-separated)")


class GenerateResponse(BaseModel):
    """Response model for wiki generation job creation."""
    job_id: str
    status: str
    status_url: str
    result_url: str
    progress_url: str


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    progress: int
    message: str
    current_page: Optional[str] = None
    pages_completed: int
    pages_total: int
    error: Optional[str] = None
    created_at: str
    updated_at: str


@router.post("/generate", response_model=GenerateResponse)
async def generate_wiki(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Start server-side wiki generation.

    This endpoint initiates wiki generation and returns immediately with job tracking URLs.
    Use the status_url to poll for progress and result_url to fetch the completed wiki.

    Example:
        POST /api/generate
        {
            "repo_url": "https://github.com/owner/repo",
            "format": "markdown"
        }

    Returns:
        GenerateResponse with job_id and URLs for status/result/progress
    """
    # Validate format
    if request.format not in ("markdown", "json"):
        raise HTTPException(status_code=400, detail="format must be 'markdown' or 'json'")

    # Validate provider
    if request.provider not in ("vertex", "google", "openai", "anthropic"):
        raise HTTPException(status_code=400, detail="Unsupported provider. Use: vertex, google, openai, anthropic")

    # Create job
    job = await job_manager.create_job(
        repo_url=request.repo_url,
        branch=request.branch,
        token=request.token,
        format=request.format,
        language=request.language,
        provider=request.provider,
        model=request.model or "gemini-2.5-flash",
        comprehensive=request.comprehensive,
        excluded_dirs=request.excluded_dirs,
        included_files=request.included_files
    )

    # Start generation in background
    generator = WikiGenerator(job_manager)
    background_tasks.add_task(generator.generate_wiki, job)

    logger.info(f"Started wiki generation job {job.id} for {request.repo_url}")

    return GenerateResponse(
        job_id=job.id,
        status=job.status.value,
        status_url=f"/api/generate/{job.id}/status",
        result_url=f"/api/generate/{job.id}/result",
        progress_url=f"/api/generate/{job.id}/progress"
    )


@router.get("/generate/{job_id}/status", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """
    Get current status of a wiki generation job.

    Use this endpoint to poll for job progress.

    Returns:
        JobStatusResponse with current job state
    """
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(**job.to_status_dict())


@router.get("/generate/{job_id}/result")
async def get_result(job_id: str):
    """
    Get the result of a completed wiki generation job.

    Returns:
        - For format=markdown: Plain text markdown document
        - For format=json: WikiCacheData JSON structure

    Raises:
        404: Job not found
        202: Job still in progress (Retry-After header set)
        500: Job failed with error
    """
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == JobStatus.ERROR:
        raise HTTPException(status_code=500, detail=job.error or "Job failed")

    if job.status != JobStatus.COMPLETE:
        raise HTTPException(
            status_code=202,
            detail=f"Job still in progress: {job.status.value}",
            headers={"Retry-After": "5"}
        )

    # Return result in appropriate format
    if job.format == "markdown":
        return StreamingResponse(
            iter([job.result]),
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=wiki.md"}
        )
    else:
        return job.result


@router.get("/generate/{job_id}/progress")
async def get_progress(job_id: str):
    """
    Stream progress updates via Server-Sent Events.

    Events:
        - progress: {status, progress, message, current_page, pages_completed, pages_total}
        - complete: Job finished successfully
        - error: Job failed with error message

    Example:
        GET /api/generate/uuid-xxx/progress

        event: progress
        data: {"status": "generating_pages", "progress": 50, "message": "Generating: Overview"}

        event: complete
        data: {"status": "complete", "progress": 100}
    """
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        queue = await job_manager.subscribe(job_id)
        try:
            # Send current state immediately
            current_status = job.to_status_dict()
            yield f"event: progress\ndata: {json.dumps(current_status)}\n\n"

            # If already complete or errored, send final event and return
            if job.status == JobStatus.COMPLETE:
                yield f"event: complete\ndata: {json.dumps(current_status)}\n\n"
                return
            elif job.status == JobStatus.ERROR:
                yield f"event: error\ndata: {json.dumps(current_status)}\n\n"
                return

            # Stream updates
            while True:
                try:
                    progress = await asyncio.wait_for(queue.get(), timeout=30.0)

                    if progress["status"] == "complete":
                        yield f"event: complete\ndata: {json.dumps(progress)}\n\n"
                        break
                    elif progress["status"] == "error":
                        yield f"event: error\ndata: {json.dumps(progress)}\n\n"
                        break
                    else:
                        yield f"event: progress\ndata: {json.dumps(progress)}\n\n"

                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f": keepalive\n\n"

        finally:
            await job_manager.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.get("/generate/jobs")
async def list_jobs():
    """
    List all active wiki generation jobs.

    Returns:
        List of job status objects
    """
    return job_manager.list_jobs()
