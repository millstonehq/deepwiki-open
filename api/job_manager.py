"""
Job manager for async wiki generation.

Provides in-memory job queue with status tracking for long-running
wiki generation tasks.
"""

import asyncio
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a wiki generation job."""
    QUEUED = "queued"
    CLONING = "cloning"
    ANALYZING = "analyzing"
    GENERATING_STRUCTURE = "generating_structure"
    GENERATING_PAGES = "generating_pages"
    CACHING = "caching"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class WikiGenerationJob:
    """Represents a wiki generation job."""
    id: str
    repo_url: str
    branch: str
    token: Optional[str]
    format: str  # "markdown" or "json"
    language: str
    provider: str
    model: str
    comprehensive: bool
    excluded_dirs: Optional[str] = None
    included_files: Optional[str] = None

    # Status tracking
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    message: str = ""
    current_page: Optional[str] = None
    pages_completed: int = 0
    pages_total: int = 0

    # Results
    result: Optional[Any] = None
    error: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # SSE subscribers
    subscribers: List[asyncio.Queue] = field(default_factory=list)

    def to_status_dict(self) -> Dict[str, Any]:
        """Convert job status to dictionary for API response."""
        return {
            "job_id": self.id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "current_page": self.current_page,
            "pages_completed": self.pages_completed,
            "pages_total": self.pages_total,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class JobManager:
    """
    In-memory job manager for wiki generation.

    Handles job creation, status updates, and subscriber notifications.
    Jobs are stored in memory and cleaned up after TTL expires.
    """

    def __init__(self, max_jobs: int = 100, job_ttl_seconds: int = 3600):
        self._jobs: Dict[str, WikiGenerationJob] = {}
        self._max_jobs = max_jobs
        self._job_ttl = job_ttl_seconds
        self._lock = asyncio.Lock()

    async def create_job(
        self,
        repo_url: str,
        branch: str = "main",
        token: Optional[str] = None,
        format: str = "markdown",
        language: str = "en",
        provider: str = "vertex",
        model: str = "gemini-2.5-flash",
        comprehensive: bool = True,
        excluded_dirs: Optional[str] = None,
        included_files: Optional[str] = None
    ) -> WikiGenerationJob:
        """Create a new wiki generation job."""
        async with self._lock:
            # Cleanup old jobs if at capacity
            await self._cleanup_old_jobs()

            job_id = str(uuid.uuid4())
            job = WikiGenerationJob(
                id=job_id,
                repo_url=repo_url,
                branch=branch,
                token=token,
                format=format,
                language=language,
                provider=provider,
                model=model,
                comprehensive=comprehensive,
                excluded_dirs=excluded_dirs,
                included_files=included_files
            )
            self._jobs[job_id] = job
            logger.info(f"Created job {job_id} for {repo_url}")
            return job

    async def get_job(self, job_id: str) -> Optional[WikiGenerationJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    async def update_job(self, job_id: str, **updates) -> None:
        """Update job status and notify subscribers."""
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Attempted to update non-existent job {job_id}")
            return

        # Apply updates
        for key, value in updates.items():
            if hasattr(job, key):
                setattr(job, key, value)
        job.updated_at = datetime.now()

        logger.debug(f"Job {job_id}: {job.status.value} - {job.message}")

        # Notify SSE subscribers
        status_dict = job.to_status_dict()
        for queue in job.subscribers:
            try:
                await queue.put(status_dict)
            except Exception as e:
                logger.warning(f"Failed to notify subscriber: {e}")

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to job progress updates via SSE."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        queue: asyncio.Queue = asyncio.Queue()
        job.subscribers.append(queue)
        logger.debug(f"Added subscriber to job {job_id}")
        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribe from job progress updates."""
        job = self._jobs.get(job_id)
        if job and queue in job.subscribers:
            job.subscribers.remove(queue)
            logger.debug(f"Removed subscriber from job {job_id}")

    async def _cleanup_old_jobs(self) -> None:
        """Remove old completed/errored jobs past TTL."""
        now = datetime.now()
        to_remove = []

        for job_id, job in self._jobs.items():
            if job.status in (JobStatus.COMPLETE, JobStatus.ERROR):
                age = (now - job.updated_at).total_seconds()
                if age > self._job_ttl:
                    to_remove.append(job_id)

        for job_id in to_remove:
            del self._jobs[job_id]
            logger.debug(f"Cleaned up old job {job_id}")

        # Also cleanup if over capacity
        if len(self._jobs) >= self._max_jobs:
            # Remove oldest completed jobs first
            completed = [
                (jid, j) for jid, j in self._jobs.items()
                if j.status in (JobStatus.COMPLETE, JobStatus.ERROR)
            ]
            completed.sort(key=lambda x: x[1].updated_at)

            while len(self._jobs) >= self._max_jobs and completed:
                job_id, _ = completed.pop(0)
                del self._jobs[job_id]
                logger.debug(f"Removed job {job_id} due to capacity")

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all active jobs."""
        return [job.to_status_dict() for job in self._jobs.values()]


# Global job manager instance
job_manager = JobManager()
