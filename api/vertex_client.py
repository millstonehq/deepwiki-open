"""Vertex AI Generative Model Client integration.

Uses Google Cloud's Vertex AI with Application Default Credentials (ADC).
Supports Workload Identity on GKE for zero-secret deployments.
"""

import os
import logging
import backoff
from typing import Dict, Any, Optional, Generator

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, GeneratorOutput

try:
    from vertexai.generative_models import GenerativeModel, Content, Part
    import vertexai
except ImportError:
    raise ImportError(
        "google-cloud-aiplatform is required. Install with 'pip install google-cloud-aiplatform'"
    )

log = logging.getLogger(__name__)


class VertexAIClient(ModelClient):
    """Vertex AI Generative Model client using Application Default Credentials.

    Uses ADC for authentication, which supports:
    - GKE Workload Identity (recommended for production)
    - Service account JSON key via GOOGLE_APPLICATION_CREDENTIALS
    - User credentials via `gcloud auth application-default login`

    Args:
        project_id: GCP project ID. Defaults to GOOGLE_CLOUD_PROJECT env var.
        location: GCP region. Defaults to VERTEX_AI_LOCATION or "us-central1".

    Example:
        ```python
        from api.vertex_client import VertexAIClient
        import adalflow as adal

        client = VertexAIClient(project_id="my-project")
        generator = adal.Generator(
            model_client=client,
            model_kwargs={
                "model": "gemini-3-flash-preview",
                "temperature": 1.0,
                "thinking_level": "medium"
            }
        )
        ```

    References:
        - Vertex AI Gemini: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
        - Available models: gemini-3-flash-preview, gemini-3-pro-preview, gemini-2.5-flash, gemini-2.5-pro
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
        self._model_cache: Dict[str, GenerativeModel] = {}

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

    def _get_model(self, model_name: str) -> GenerativeModel:
        """Get or create a cached model instance."""
        if model_name not in self._model_cache:
            self._model_cache[model_name] = GenerativeModel(model_name)
        return self._model_cache[model_name]

    def parse_chat_completion(self, response) -> GeneratorOutput:
        """Parse Vertex AI response to GeneratorOutput format."""
        try:
            # Extract text from response
            if hasattr(response, 'text'):
                text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                text = response.candidates[0].content.parts[0].text
            else:
                text = str(response)

            return GeneratorOutput(
                data=text,
                error=None,
                raw_response=response
            )
        except Exception as e:
            log.error(f"Error parsing Vertex AI response: {e}")
            return GeneratorOutput(
                data=None,
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
        final_kwargs = model_kwargs.copy()

        # Handle different input types
        if isinstance(input, str):
            final_kwargs["prompt"] = input
        elif isinstance(input, list):
            # Assume it's a list of messages for chat
            final_kwargs["messages"] = input
        elif input is not None:
            final_kwargs["prompt"] = str(input)

        # Set default model
        if "model" not in final_kwargs:
            final_kwargs["model"] = "gemini-3-flash-preview"

        return final_kwargs

    def _build_contents(self, messages: list) -> list:
        """Convert messages to Vertex AI Content format."""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            # Vertex AI uses "user" and "model" roles
            if role == "assistant":
                role = "model"
            elif role == "system":
                # System messages are typically prepended to user message
                role = "user"

            text = msg.get("content", "")
            contents.append(Content(role=role, parts=[Part.from_text(text)]))
        return contents

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_time=30,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Call Vertex AI generative API."""
        self._ensure_initialized()

        model_name = api_kwargs.get("model", "gemini-3-flash-preview")
        prompt = api_kwargs.get("prompt")
        messages = api_kwargs.get("messages")

        # Build generation config from remaining kwargs
        gen_config_keys = ["temperature", "top_p", "top_k", "max_output_tokens", "candidate_count", "thinking_level"]
        generation_config = {k: v for k, v in api_kwargs.items() if k in gen_config_keys}

        log.debug(f"Vertex AI call: model={model_name}, gen_config={generation_config}")

        try:
            model = self._get_model(model_name)

            if messages:
                contents = self._build_contents(messages)
                response = model.generate_content(
                    contents,
                    generation_config=generation_config if generation_config else None
                )
            elif prompt:
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config if generation_config else None
                )
            else:
                raise ValueError("Either 'prompt' or 'messages' must be provided")

            return response

        except Exception as e:
            log.error(f"Error calling Vertex AI: {e}")
            raise

    async def acall(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Async call - Vertex AI supports async natively."""
        self._ensure_initialized()

        model_name = api_kwargs.get("model", "gemini-3-flash-preview")
        prompt = api_kwargs.get("prompt")
        messages = api_kwargs.get("messages")

        gen_config_keys = ["temperature", "top_p", "top_k", "max_output_tokens", "candidate_count", "thinking_level"]
        generation_config = {k: v for k, v in api_kwargs.items() if k in gen_config_keys}

        try:
            model = self._get_model(model_name)

            if messages:
                contents = self._build_contents(messages)
                response = await model.generate_content_async(
                    contents,
                    generation_config=generation_config if generation_config else None
                )
            elif prompt:
                response = await model.generate_content_async(
                    prompt,
                    generation_config=generation_config if generation_config else None
                )
            else:
                raise ValueError("Either 'prompt' or 'messages' must be provided")

            return response

        except Exception as e:
            log.error(f"Error calling Vertex AI async: {e}")
            raise
