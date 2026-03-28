"""
FAL provider implementation.

This module implements the BaseProvider interface for FAL AI services,
supporting all model capabilities by dynamically discovering FAL nodes
and executing models through their node implementations.
"""

import ast
import importlib
import inspect
import json
import os
import pkgutil
import re
from collections.abc import AsyncGenerator
from typing import Any, List

import numpy as np
from nodetool.providers.base import BaseProvider
from nodetool.providers.types import (
    ImageBytes,
    TextToImageParams,
    ImageToImageParams,
    VideoBytes,
    TextToVideoParams,
    ImageToVideoParams,
    Model3DBytes,
    TextTo3DParams,
    ImageTo3DParams,
)
from nodetool.config.environment import Environment
from nodetool.metadata.types import (
    ASRModel,
    ImageModel,
    Model3DModel,
    Provider,
    TTSModel,
    VideoModel,
)
from nodetool.workflows.base_node import ApiKeyMissingError
from fal_client import AsyncClient
import httpx
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

# Module name -> capability category mapping
_MODULE_CATEGORY: dict[str, str] = {
    "text_to_image": "image",
    "image_to_image": "image",
    "text_to_video": "video",
    "image_to_video": "video",
    "audio_to_video": "video",
    "video_to_video": "video",
    "text_to_audio": "tts",
    "text_to_speech": "tts",
    "speech_to_text": "asr",
    "audio_to_text": "asr",
    "text_to_3d": "3d",
    "image_to_3d": "3d",
    "model3d": "3d",
    "3d_to_3d": "3d",
}


def _get_endpoint_id(cls: type) -> str | None:
    """Extract the FAL endpoint ID from a node class by parsing its source.

    Looks for the ``application`` keyword in ``submit_request`` calls.
    """
    try:
        source = inspect.getsource(cls)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if hasattr(func, "attr") and func.attr == "submit_request":
                    for kw in node.keywords:
                        if kw.arg == "application" and isinstance(
                            kw.value, ast.Constant
                        ):
                            return kw.value.value
    except Exception:
        pass
    return None


def _get_node_name(cls: type) -> str:
    """Derive a human-readable model name from a node class."""
    doc = (cls.__doc__ or "").strip()
    if doc:
        first_line = doc.split("\n")[0].strip()
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."
        return first_line
    return cls.__name__


def _node_class_path(cls: type) -> str:
    """Return the fully qualified import path for a node class."""
    return f"{cls.__module__}.{cls.__name__}"


def _discover_fal_nodes() -> dict[str, list[tuple[str, str, str]]]:
    """Discover all FALNode subclasses from node modules.

    Returns a dict mapping category to list of (endpoint_id, name, class_path) tuples.
    Categories: ``image``, ``video``, ``tts``, ``asr``, ``3d``.
    """
    from nodetool.nodes.fal.fal_node import FALNode

    result: dict[str, list[tuple[str, str, str]]] = {
        "image": [],
        "video": [],
        "tts": [],
        "asr": [],
        "3d": [],
    }

    import nodetool.nodes.fal as fal_pkg

    skip = {
        "nodetool.nodes.fal.fal_node",
        "nodetool.nodes.fal.types",
        "nodetool.nodes.fal.__init__",
        "nodetool.nodes.fal.dynamic_schema",
    }

    for _importer, modname, ispkg in pkgutil.iter_modules(
        fal_pkg.__path__, fal_pkg.__name__ + "."
    ):
        if ispkg or modname in skip:
            continue
        short = modname.rsplit(".", 1)[-1]
        category = _MODULE_CATEGORY.get(short)
        if category is None:
            continue
        try:
            mod = importlib.import_module(modname)
        except Exception:
            log.debug("Failed to import FAL module %s", modname)
            continue

        for _name, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(obj, FALNode)
                and obj is not FALNode
                and obj.__module__ == mod.__name__
            ):
                endpoint = _get_endpoint_id(obj)
                if endpoint:
                    result[category].append(
                        (endpoint, _get_node_name(obj), _node_class_path(obj))
                    )
    return result


# Cache discovered nodes at module level (populated on first access)
_discovered_nodes: dict[str, list[tuple[str, str, str]]] | None = None


def _get_discovered_nodes() -> dict[str, list[tuple[str, str, str]]]:
    global _discovered_nodes
    if _discovered_nodes is None:
        _discovered_nodes = _discover_fal_nodes()
    return _discovered_nodes


def _find_node_class(model_id: str, category: str | None = None) -> type | None:
    """Find the FAL node class for a given model/endpoint ID.

    Args:
        model_id: The FAL endpoint ID (e.g. ``fal-ai/flux/dev``).
        category: Optional category to narrow the search.

    Returns:
        The node class, or ``None`` if not found.
    """
    nodes = _get_discovered_nodes()
    categories = [category] if category else list(nodes.keys())
    for cat in categories:
        for endpoint, _name, class_path in nodes.get(cat, []):
            if endpoint == model_id:
                module_path, cls_name = class_path.rsplit(".", 1)
                mod = importlib.import_module(module_path)
                return getattr(mod, cls_name, None)
    return None


class FalProvider(BaseProvider):
    """FAL AI provider supporting all model capabilities.

    Capabilities are discovered dynamically by introspecting FAL node modules.
    All models are executed via their corresponding node implementations.
    """

    provider_name = "fal_ai"

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["FAL_API_KEY"]

    def __init__(self, secrets: dict[str, str] | None = None):
        super().__init__(secrets=secrets)
        self.api_key = (secrets or {}).get(
            "FAL_API_KEY"
        ) or Environment.get_environment().get("FAL_API_KEY")
        if not self.api_key:
            raise ApiKeyMissingError("FAL_API_KEY is not configured")
        # Set FAL_KEY environment variable for the client
        os.environ["FAL_KEY"] = self.api_key

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables needed when running inside Docker."""
        return {"FAL_API_KEY": self.api_key} if self.api_key else {}

    def _get_client(self) -> AsyncClient:
        """Get the FAL async client."""
        return AsyncClient()

    @staticmethod
    def _format_validation_error(error_str: str) -> str:
        """Format FAL validation errors into user-friendly messages.

        Args:
            error_str: The error string from FAL API

        Returns:
            A formatted, human-readable error message
        """
        # Try to parse validation error format
        # Example: [{'type': 'less_than_equal', 'loc': ['body', 'num_inference_steps'], 'msg': '...', 'input': 30, 'ctx': {'le': 12}}]

        try:
            # Try to extract the validation error list from the string
            if "[{" in error_str and "}]" in error_str:
                # Extract the list portion
                match = re.search(r"\[{.*}\]", error_str, re.DOTALL)
                if match:
                    error_list_str = match.group(0)

                    # Try JSON parsing first (for properly formatted JSON with double quotes)
                    try:
                        errors = json.loads(error_list_str)
                    except json.JSONDecodeError:
                        # If JSON fails, use ast.literal_eval for Python dict format (single quotes)
                        errors = ast.literal_eval(error_list_str)

                    if isinstance(errors, list) and errors:
                        formatted_errors = []
                        for error in errors:
                            if isinstance(error, dict):
                                # Extract field name from location
                                loc = error.get("loc", [])
                                field_name = loc[-1] if loc else "unknown field"

                                # Get the actual error message
                                msg = error.get("msg", "")
                                input_value = error.get("input")
                                ctx = error.get("ctx", {})

                                # Format based on error type
                                error_type = error.get("type", "")

                                if error_type == "less_than_equal" and "le" in ctx:
                                    formatted_errors.append(
                                        f"Parameter '{field_name}' must be {ctx['le']} or less (you provided {input_value})"
                                    )
                                elif error_type == "greater_than_equal" and "ge" in ctx:
                                    formatted_errors.append(
                                        f"Parameter '{field_name}' must be {ctx['ge']} or greater (you provided {input_value})"
                                    )
                                elif error_type == "missing":
                                    formatted_errors.append(
                                        f"Required parameter '{field_name}' is missing"
                                    )
                                elif error_type == "value_error":
                                    formatted_errors.append(
                                        f"Parameter '{field_name}': {msg}"
                                    )
                                elif input_value is not None:
                                    formatted_errors.append(
                                        f"Parameter '{field_name}': {msg} (you provided {input_value})"
                                    )
                                else:
                                    formatted_errors.append(
                                        f"Parameter '{field_name}': {msg}"
                                    )

                        if formatted_errors:
                            return "Invalid parameters:\n  - " + "\n  - ".join(
                                formatted_errors
                            )
        except (json.JSONDecodeError, ValueError, SyntaxError, KeyError, TypeError):
            # If parsing fails, fall through to return original error
            pass

        # Return original error if we couldn't parse it
        return error_str

    # ------------------------------------------------------------------
    # Model discovery – built dynamically from FAL node modules
    # ------------------------------------------------------------------

    async def get_available_image_models(self) -> List[ImageModel]:
        """Get available FAL AI image models by introspecting node modules."""
        nodes = _get_discovered_nodes()
        return [
            ImageModel(
                id=endpoint,
                name=name,
                provider=Provider.FalAI,
                path=class_path,
            )
            for endpoint, name, class_path in nodes.get("image", [])
        ]

    async def get_available_video_models(self) -> list[VideoModel]:
        """Get available FAL AI video models by introspecting node modules."""
        nodes = _get_discovered_nodes()
        return [
            VideoModel(
                id=endpoint,
                name=name,
                provider=Provider.FalAI,
                path=class_path,
            )
            for endpoint, name, class_path in nodes.get("video", [])
        ]

    async def get_available_tts_models(self) -> list[TTSModel]:
        """Get available FAL AI text-to-speech models by introspecting node modules."""
        nodes = _get_discovered_nodes()
        return [
            TTSModel(
                id=endpoint,
                name=name,
                provider=Provider.FalAI,
                path=class_path,
            )
            for endpoint, name, class_path in nodes.get("tts", [])
        ]

    async def get_available_asr_models(self) -> list[ASRModel]:
        """Get available FAL AI ASR models by introspecting node modules."""
        nodes = _get_discovered_nodes()
        return [
            ASRModel(
                id=endpoint,
                name=name,
                provider=Provider.FalAI,
                path=class_path,
            )
            for endpoint, name, class_path in nodes.get("asr", [])
        ]

    async def get_available_3d_models(self) -> list[Model3DModel]:
        """Get available FAL AI 3D generation models by introspecting node modules."""
        nodes = _get_discovered_nodes()
        return [
            Model3DModel(
                id=endpoint,
                name=name,
                provider=Provider.FalAI,
                path=class_path,
            )
            for endpoint, name, class_path in nodes.get("3d", [])
        ]

    # ------------------------------------------------------------------
    # Capability methods – execute via FAL nodes
    # ------------------------------------------------------------------

    async def execute_via_node(
        self,
        model_id: str,
        context: ProcessingContext,
        category: str | None = None,
        **field_overrides: Any,
    ) -> Any:
        """Execute any discovered FAL model through its node implementation.

        This is the primary entry-point for running FAL models via nodes.
        The *model_id* is resolved to the corresponding ``FALNode`` subclass,
        an instance is created with the given *field_overrides*, and
        ``process()`` is called.

        Args:
            model_id: FAL endpoint ID (e.g. ``"fal-ai/flux/dev"``).
            context: Processing context with secrets.
            category: Optional category hint (``"image"``, ``"video"``, etc.)
                for faster lookup.
            **field_overrides: Field values to set on the node before processing.

        Returns:
            The result of the node's ``process`` method.

        Raises:
            ValueError: If no node is found for the given *model_id*.
        """
        node_cls = _find_node_class(model_id, category=category)
        if node_cls is None:
            raise ValueError(f"No FAL node found for model '{model_id}'")
        node = node_cls(**field_overrides)
        return await node.process(context)

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
    ) -> ImageBytes:
        """Generate an image from a text prompt using FAL AI.

        Args:
            params: Text-to-image generation parameters
            timeout_s: Optional timeout in seconds

        Returns:
            Raw image bytes as PNG

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        client = self._get_client()

        # Build arguments for FAL API
        arguments: dict[str, Any] = {
            "prompt": params.prompt,
            "output_format": "png",
        }

        # Add optional parameters if provided
        if params.negative_prompt:
            arguments["negative_prompt"] = params.negative_prompt
        if params.guidance_scale is not None:
            arguments["guidance_scale"] = params.guidance_scale
        if params.num_inference_steps is not None:
            arguments["num_inference_steps"] = params.num_inference_steps
        if params.width and params.height:
            # FAL uses image_size for some models
            arguments["image_size"] = {"width": params.width, "height": params.height}
        if params.seed is not None and params.seed != -1:
            arguments["seed"] = params.seed
        if params.safety_check is not None:
            arguments["enable_safety_checker"] = params.safety_check

        try:
            # Submit request and wait for result
            handler = await client.submit(params.model.id, arguments=arguments)
            result = await handler.get()

            # Extract image URL from result
            if "images" in result and len(result["images"]) > 0:
                image_url = result["images"][0]["url"]
            elif "image" in result:
                image_url = result["image"]["url"]
            else:
                raise RuntimeError(f"Unexpected FAL response format: {result}")

            # Download the image
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(image_url)
                response.raise_for_status()
                image_bytes = response.content

            return image_bytes

        except Exception as e:
            error_msg = self._format_validation_error(str(e))
            raise RuntimeError(f"FAL text-to-image generation failed: {error_msg}")

    async def image_to_image(
        self,
        image: ImageBytes,
        params: ImageToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
    ) -> ImageBytes:
        """Transform an image based on a text prompt using FAL AI.

        Args:
            image: Input image as bytes
            params: Image-to-image generation parameters
            timeout_s: Optional timeout in seconds

        Returns:
            Raw image bytes as PNG

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        client = self._get_client()

        # FAL requires images as data URIs for image-to-image
        import base64

        image_b64 = base64.b64encode(image).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{image_b64}"

        # Build arguments for FAL API
        arguments: dict[str, Any] = {
            "prompt": params.prompt,
            "image_url": image_data_uri,
            "output_format": "png",
        }

        # Add optional parameters if provided
        if params.negative_prompt:
            arguments["negative_prompt"] = params.negative_prompt
        if params.guidance_scale is not None:
            arguments["guidance_scale"] = params.guidance_scale
        if params.num_inference_steps is not None:
            arguments["num_inference_steps"] = params.num_inference_steps
        if params.strength is not None:
            arguments["strength"] = params.strength
        if params.target_width and params.target_height:
            arguments["image_size"] = {
                "width": params.target_width,
                "height": params.target_height,
            }
        if params.seed is not None and params.seed != -1:
            arguments["seed"] = params.seed

        try:
            # Submit request and wait for result
            handler = await client.submit(params.model.id, arguments=arguments)
            result = await handler.get()

            # Extract image URL from result
            if "images" in result and len(result["images"]) > 0:
                image_url = result["images"][0]["url"]
            elif "image" in result:
                image_url = result["image"]["url"]
            else:
                raise RuntimeError(f"Unexpected FAL response format: {result}")

            # Download the image
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(image_url)
                response.raise_for_status()
                image_bytes = response.content

            return image_bytes

        except Exception as e:
            error_msg = self._format_validation_error(str(e))
            raise RuntimeError(f"FAL image-to-image generation failed: {error_msg}")

    async def text_to_speech(
        self,
        text: str,
        model: str,
        voice: str | None = None,
        speed: float = 1.0,
        timeout_s: int | None = None,
        context: Any = None,  # ProcessingContext, but imported later
        **kwargs: Any,
    ) -> AsyncGenerator[np.ndarray[Any, np.dtype[np.int16]], None]:
        """Generate speech audio from text using FAL AI text-to-audio models.

        Args:
            text: Input text to convert to speech
            model: FAL model identifier (e.g., "fal-ai/mmaudio-v2/text-to-audio")
            voice: Voice identifier (not used by most FAL TTS models)
            speed: Speech speed multiplier (not used by most FAL TTS models)
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            **kwargs: Additional FAL parameters (e.g., num_steps, duration, cfg_strength)

        Returns:
            Raw audio bytes (typically FLAC or WAV format)

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(f"Generating speech with FAL model: {model}")

        if not text:
            raise ValueError("text must not be empty")

        client = self._get_client()

        # Build arguments for FAL API
        arguments: dict[str, Any] = {
            "prompt": text,
        }

        # Add optional parameters from kwargs
        if "num_steps" in kwargs:
            arguments["num_steps"] = kwargs["num_steps"]
        if "duration" in kwargs:
            arguments["duration"] = kwargs["duration"]
        if "cfg_strength" in kwargs:
            arguments["cfg_strength"] = kwargs["cfg_strength"]
        if "negative_prompt" in kwargs:
            arguments["negative_prompt"] = kwargs["negative_prompt"]
        if "seed" in kwargs and kwargs["seed"] != -1:
            arguments["seed"] = kwargs["seed"]

        try:
            # Submit request and wait for result
            handler = await client.submit(model, arguments=arguments)
            result = await handler.get()

            # Extract audio URL from result
            if "audio" in result:
                audio_url = result["audio"]["url"]
            else:
                raise RuntimeError(f"Unexpected FAL response format: {result}")

            # Download the audio
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(audio_url)
                response.raise_for_status()
                audio_bytes = response.content
                yield np.frombuffer(audio_bytes, dtype=np.int16)
                log.debug(f"Generated {len(audio_bytes)} bytes of audio")

        except Exception as e:
            error_msg = self._format_validation_error(str(e))
            raise RuntimeError(f"FAL text-to-speech generation failed: {error_msg}")

    async def automatic_speech_recognition(
        self,
        audio: bytes,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        timeout_s: int | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> str:
        """Transcribe audio to text using a FAL ASR model.

        Args:
            audio: Input audio as bytes.
            model: FAL model identifier (endpoint ID).
            language: Optional language code.
            prompt: Optional guiding prompt.
            temperature: Sampling temperature.
            timeout_s: Optional timeout in seconds.
            context: Optional processing context.

        Returns:
            Transcribed text.
        """
        client = self._get_client()

        # Upload audio so FAL can access it
        audio_url = await client.upload(audio, "audio/mp3")

        arguments: dict[str, Any] = {"audio_url": audio_url}
        if language:
            arguments["language"] = language
        if prompt:
            arguments["prompt"] = prompt
        if temperature != 0.0:
            arguments["temperature"] = temperature

        try:
            handler = await client.submit(model, arguments=arguments)
            result = await handler.get()

            # Different ASR models return text in different fields
            if "text" in result:
                return result["text"]
            if "transcription" in result:
                return result["transcription"]
            # Fallback: return the whole result as string
            return str(result)

        except Exception as e:
            error_msg = self._format_validation_error(str(e))
            raise RuntimeError(f"FAL ASR failed: {error_msg}")

    async def text_to_video(
        self,
        params: TextToVideoParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> VideoBytes:
        """Generate a video from a text prompt using a FAL model.

        Args:
            params: Text-to-video generation parameters.
            timeout_s: Optional timeout in seconds.
            context: Optional processing context.
            node_id: Optional node ID for tracking.

        Returns:
            Raw video bytes.
        """
        client = self._get_client()

        arguments: dict[str, Any] = {"prompt": params.prompt}
        if params.negative_prompt:
            arguments["negative_prompt"] = params.negative_prompt
        if params.aspect_ratio:
            arguments["aspect_ratio"] = params.aspect_ratio
        if params.resolution:
            arguments["resolution"] = params.resolution
        if params.guidance_scale is not None:
            arguments["guidance_scale"] = params.guidance_scale
        if params.num_inference_steps is not None:
            arguments["num_inference_steps"] = params.num_inference_steps
        if params.seed is not None and params.seed != -1:
            arguments["seed"] = params.seed

        try:
            handler = await client.submit(params.model.id, arguments=arguments)
            result = await handler.get()

            # Extract video URL
            if "video" in result:
                video_url = result["video"]["url"]
            elif "videos" in result and len(result["videos"]) > 0:
                video_url = result["videos"][0]["url"]
            else:
                raise RuntimeError(f"Unexpected FAL response format: {result}")

            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(video_url)
                response.raise_for_status()
                return response.content

        except Exception as e:
            error_msg = self._format_validation_error(str(e))
            raise RuntimeError(f"FAL text-to-video generation failed: {error_msg}")

    async def image_to_video(
        self,
        image: bytes,
        params: ImageToVideoParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
        **kwargs: Any,
    ) -> VideoBytes:
        """Generate a video from an input image using a FAL model.

        Args:
            image: Input image as bytes.
            params: Image-to-video generation parameters.
            timeout_s: Optional timeout in seconds.
            context: Optional processing context.
            node_id: Optional node ID for tracking.

        Returns:
            Raw video bytes.
        """
        import base64

        client = self._get_client()

        image_b64 = base64.b64encode(image).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{image_b64}"

        arguments: dict[str, Any] = {"image_url": image_data_uri}
        if params.prompt:
            arguments["prompt"] = params.prompt
        if params.negative_prompt:
            arguments["negative_prompt"] = params.negative_prompt
        if params.aspect_ratio:
            arguments["aspect_ratio"] = params.aspect_ratio
        if params.resolution:
            arguments["resolution"] = params.resolution
        if params.guidance_scale is not None:
            arguments["guidance_scale"] = params.guidance_scale
        if params.num_inference_steps is not None:
            arguments["num_inference_steps"] = params.num_inference_steps
        if params.seed is not None and params.seed != -1:
            arguments["seed"] = params.seed

        try:
            handler = await client.submit(params.model.id, arguments=arguments)
            result = await handler.get()

            if "video" in result:
                video_url = result["video"]["url"]
            elif "videos" in result and len(result["videos"]) > 0:
                video_url = result["videos"][0]["url"]
            else:
                raise RuntimeError(f"Unexpected FAL response format: {result}")

            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(video_url)
                response.raise_for_status()
                return response.content

        except Exception as e:
            error_msg = self._format_validation_error(str(e))
            raise RuntimeError(f"FAL image-to-video generation failed: {error_msg}")

    async def text_to_3d(
        self,
        params: TextTo3DParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> Model3DBytes:
        """Generate a 3D model from a text prompt using a FAL model.

        Args:
            params: Text-to-3D generation parameters.
            timeout_s: Optional timeout in seconds.
            context: Optional processing context.
            node_id: Optional node ID for tracking.

        Returns:
            Raw 3D model bytes (GLB, OBJ, etc.).
        """
        client = self._get_client()

        arguments: dict[str, Any] = {"prompt": params.prompt}
        if params.negative_prompt:
            arguments["negative_prompt"] = params.negative_prompt
        if params.seed is not None and params.seed != -1:
            arguments["seed"] = params.seed

        try:
            handler = await client.submit(params.model.id, arguments=arguments)
            result = await handler.get()

            # Try common 3D output field names
            model_url = None
            for key in ("glb", "model", "output", "model_mesh"):
                if key in result:
                    val = result[key]
                    if isinstance(val, dict) and "url" in val:
                        model_url = val["url"]
                        break
                    elif isinstance(val, str):
                        model_url = val
                        break

            if model_url is None:
                raise RuntimeError(f"Unexpected FAL response format: {result}")

            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(model_url)
                response.raise_for_status()
                return response.content

        except Exception as e:
            error_msg = self._format_validation_error(str(e))
            raise RuntimeError(f"FAL text-to-3D generation failed: {error_msg}")

    async def image_to_3d(
        self,
        image: bytes,
        params: ImageTo3DParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> Model3DBytes:
        """Generate a 3D model from an input image using a FAL model.

        Args:
            image: Input image as bytes.
            params: Image-to-3D generation parameters.
            timeout_s: Optional timeout in seconds.
            context: Optional processing context.
            node_id: Optional node ID for tracking.

        Returns:
            Raw 3D model bytes (GLB, OBJ, etc.).
        """
        import base64

        client = self._get_client()

        image_b64 = base64.b64encode(image).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{image_b64}"

        arguments: dict[str, Any] = {"image_url": image_data_uri}
        if params.prompt:
            arguments["prompt"] = params.prompt
        if params.seed is not None and params.seed != -1:
            arguments["seed"] = params.seed

        try:
            handler = await client.submit(params.model.id, arguments=arguments)
            result = await handler.get()

            model_url = None
            for key in ("glb", "model", "output", "model_mesh"):
                if key in result:
                    val = result[key]
                    if isinstance(val, dict) and "url" in val:
                        model_url = val["url"]
                        break
                    elif isinstance(val, str):
                        model_url = val
                        break

            if model_url is None:
                raise RuntimeError(f"Unexpected FAL response format: {result}")

            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(model_url)
                response.raise_for_status()
                return response.content

        except Exception as e:
            error_msg = self._format_validation_error(str(e))
            raise RuntimeError(f"FAL image-to-3D generation failed: {error_msg}")
