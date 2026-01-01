"""Vertex AI Embeddings ModelClient integration.

Uses Google Cloud's Vertex AI with Application Default Credentials (ADC).
Supports Workload Identity on GKE for zero-secret deployments.
"""

import os
import logging
import backoff
from typing import Dict, Any, Optional, List, Sequence

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, EmbedderOutput

try:
    from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
    import vertexai
except ImportError:
    raise ImportError(
        "google-cloud-aiplatform is required. Install with 'pip install google-cloud-aiplatform'"
    )

log = logging.getLogger(__name__)


class VertexEmbedderClient(ModelClient):
    """Vertex AI Embeddings client using Application Default Credentials.

    Uses ADC for authentication, which supports:
    - GKE Workload Identity (recommended for production)
    - Service account JSON key via GOOGLE_APPLICATION_CREDENTIALS
    - User credentials via `gcloud auth application-default login`

    Args:
        project_id: GCP project ID. Defaults to GOOGLE_CLOUD_PROJECT env var.
        location: GCP region. Defaults to VERTEX_AI_LOCATION or "us-central1".

    Example:
        ```python
        from api.vertex_embedder_client import VertexEmbedderClient
        import adalflow as adal

        client = VertexEmbedderClient(project_id="my-project")
        embedder = adal.Embedder(
            model_client=client,
            model_kwargs={
                "model": "text-embedding-005",
                "task_type": "SEMANTIC_SIMILARITY"
            }
        )
        ```

    References:
        - Vertex AI Embeddings: https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings
        - Available models: text-embedding-005, text-embedding-004, text-multilingual-embedding-002
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
    ):
        super().__init__()
        self._project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.getenv("VERTEX_AI_LOCATION", "us-central1")
        self._initialized = False
        self._model_cache: Dict[str, TextEmbeddingModel] = {}

    def _ensure_initialized(self):
        """Initialize Vertex AI SDK lazily."""
        if self._initialized:
            return

        if not self._project_id:
            raise ValueError(
                "project_id is required. Set GOOGLE_CLOUD_PROJECT env var or pass project_id."
            )

        vertexai.init(project=self._project_id, location=self._location)
        self._initialized = True
        log.info(f"Vertex AI initialized: project={self._project_id}, location={self._location}")

    def _get_model(self, model_name: str) -> TextEmbeddingModel:
        """Get or create a cached model instance."""
        if model_name not in self._model_cache:
            self._model_cache[model_name] = TextEmbeddingModel.from_pretrained(model_name)
        return self._model_cache[model_name]

    def parse_embedding_response(self, response) -> EmbedderOutput:
        """Parse Vertex AI embedding response to EmbedderOutput format."""
        try:
            from adalflow.core.types import Embedding

            embedding_data = []
            for i, embedding in enumerate(response):
                embedding_data.append(
                    Embedding(embedding=embedding.values, index=i)
                )

            if embedding_data:
                first_dim = len(embedding_data[0].embedding) if embedding_data[0].embedding else 0
                log.info(f"Parsed {len(embedding_data)} embedding(s) (dim={first_dim})")

            return EmbedderOutput(
                data=embedding_data,
                error=None,
                raw_response=response
            )
        except Exception as e:
            log.error(f"Error parsing Vertex AI embedding response: {e}")
            return EmbedderOutput(
                data=[],
                error=str(e),
                raw_response=response
            )

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """Convert inputs to Vertex AI API format."""
        if model_type != ModelType.EMBEDDER:
            raise ValueError(f"VertexEmbedderClient only supports EMBEDDER, got {model_type}")

        if isinstance(input, str):
            texts = [input]
        elif isinstance(input, Sequence):
            texts = list(input)
        else:
            raise TypeError("input must be a string or sequence of strings")

        final_kwargs = model_kwargs.copy()
        final_kwargs["texts"] = texts

        # Set defaults
        if "model" not in final_kwargs:
            final_kwargs["model"] = "text-embedding-005"
        if "task_type" not in final_kwargs:
            final_kwargs["task_type"] = "SEMANTIC_SIMILARITY"

        return final_kwargs

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Call Vertex AI embedding API."""
        if model_type != ModelType.EMBEDDER:
            raise ValueError("VertexEmbedderClient only supports EMBEDDER model type")

        self._ensure_initialized()

        model_name = api_kwargs.get("model", "text-embedding-005")
        texts = api_kwargs.get("texts", [])
        task_type = api_kwargs.get("task_type", "SEMANTIC_SIMILARITY")

        log.debug(f"Vertex AI embedding: model={model_name}, texts={len(texts)}, task={task_type}")

        try:
            model = self._get_model(model_name)

            # Create embedding inputs with task type
            inputs = [
                TextEmbeddingInput(text=text, task_type=task_type)
                for text in texts
            ]

            embeddings = model.get_embeddings(inputs)
            return embeddings

        except Exception as e:
            log.error(f"Error calling Vertex AI Embeddings: {e}")
            raise

    async def acall(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Async call - falls back to sync for now."""
        return self.call(api_kwargs, model_type)
