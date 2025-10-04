"""
FAL image provider implementation.

This module implements the ImageProvider interface for FAL AI services.
"""

import ast
import json
import os
import re
from typing import Any, List, Set
from nodetool.image.providers.base import ImageProvider
from nodetool.chat.providers.base import ProviderCapability
from nodetool.image.types import ImageBytes, TextToImageParams, ImageToImageParams
from nodetool.config.environment import Environment
from nodetool.metadata.types import ImageModel, Provider
from nodetool.workflows.base_node import ApiKeyMissingError
from fal_client import AsyncClient
import httpx
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class FalImageProvider(ImageProvider):
    """Image provider for FAL AI services."""

    provider_name = "fal_ai"

    def __init__(self):
        super().__init__()
        env = Environment.get_environment()
        api_key = env.get("FAL_API_KEY")
        if not api_key:
            raise ApiKeyMissingError("FAL_API_KEY is not configured")
        self.api_key = api_key
        # Set FAL_KEY environment variable for the client
        os.environ["FAL_KEY"] = self.api_key

    def get_capabilities(self) -> Set[ProviderCapability]:
        """FAL provider supports both text-to-image and image-to-image generation."""
        return {
            ProviderCapability.TEXT_TO_IMAGE,
            ProviderCapability.IMAGE_TO_IMAGE,
        }

    def get_container_env(self) -> dict[str, str]:
        """Return environment variables needed when running inside Docker."""
        return {"FAL_API_KEY": self.api_key}

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

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
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
        self._log_api_request("text_to_image", params)

        client = self._get_client()

        # Build arguments for FAL API
        arguments: dict[str, Any] = {
            "prompt": params.prompt,
            "output_format": params.image_format or "png",
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

            self.usage["total_requests"] += 1
            self.usage["total_images"] += 1
            self._log_api_response("text_to_image", 1)

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
        self._log_api_request("image_to_image", params)

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

            self.usage["total_requests"] += 1
            self.usage["total_images"] += 1
            self._log_api_response("image_to_image", 1)

            return image_bytes

        except Exception as e:
            error_msg = self._format_validation_error(str(e))
            raise RuntimeError(f"FAL image-to-image generation failed: {error_msg}")

    async def get_available_image_models(self) -> List[ImageModel]:
        """
        Get available FAL AI image models.

        Returns models only if FAL_API_KEY is configured.

        Returns:
            List of ImageModel instances for FAL
        """
        env = Environment.get_environment()
        if "FAL_API_KEY" not in env:
            return []

        return [
            # FLUX Models
            ImageModel(
                id="fal-ai/flux/dev", name="FLUX.1 Dev", provider=Provider.FalAI
            ),
            ImageModel(
                id="fal-ai/flux/schnell",
                name="FLUX.1 Schnell",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/flux-pro/v1.1",
                name="FLUX.1 Pro v1.1",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/flux-pro/v1.1-ultra",
                name="FLUX.1 Pro Ultra",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/flux-pro/new",
                name="FLUX.1 Pro (New)",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/flux-lora",
                name="FLUX.1 Dev with LoRA",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/flux-subject",
                name="FLUX.1 Subject",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/flux-general",
                name="FLUX.1 General",
                provider=Provider.FalAI,
            ),
            # Ideogram Models
            ImageModel(
                id="fal-ai/ideogram/v2",
                name="Ideogram v2",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/ideogram/v2/turbo",
                name="Ideogram v2 Turbo",
                provider=Provider.FalAI,
            ),
            # Recraft Models
            ImageModel(
                id="fal-ai/recraft-v3",
                name="Recraft v3",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/recraft-20b",
                name="Recraft 20B",
                provider=Provider.FalAI,
            ),
            # Stable Diffusion Models
            ImageModel(
                id="fal-ai/stable-diffusion-v3-medium",
                name="Stable Diffusion v3 Medium",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/stable-diffusion-v35-large",
                name="Stable Diffusion v3.5 Large",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/fast-sdxl",
                name="Fast SDXL",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/stable-cascade",
                name="Stable Cascade",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/fast-lightning-sdxl",
                name="Fast Lightning SDXL",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/hyper-sdxl",
                name="Hyper SDXL",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/fast-turbo-diffusion",
                name="Fast Turbo Diffusion",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/fast-lcm-diffusion",
                name="Fast LCM Diffusion",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/lcm",
                name="LCM Diffusion",
                provider=Provider.FalAI,
            ),
            # Bria Models (Licensed Data)
            ImageModel(
                id="fal-ai/bria/text-to-image/base",
                name="Bria v1",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/bria/text-to-image/fast",
                name="Bria v1 Fast",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/bria/text-to-image/hd",
                name="Bria v1 HD",
                provider=Provider.FalAI,
            ),
            # Other Models
            ImageModel(
                id="fal-ai/aura-flow",
                name="AuraFlow v0.3",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/switti/1024",
                name="Switti",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/sana",
                name="Sana v1",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/omnigen-v1",
                name="OmniGen v1",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/luma-photon",
                name="Luma Photon",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/luma-photon/flash",
                name="Luma Photon Flash",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/playground-v25",
                name="Playground v2.5",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/fooocus",
                name="Fooocus",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/illusion-diffusion",
                name="Illusion Diffusion",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/imagen4/preview",
                name="Imagen 4 Preview",
                provider=Provider.FalAI,
            ),
            ImageModel(
                id="fal-ai/lora",
                name="LoRA Text-to-Image",
                provider=Provider.FalAI,
            ),
        ]
