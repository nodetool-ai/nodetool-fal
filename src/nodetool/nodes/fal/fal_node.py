import logging
import re
from enum import Enum
from typing import Any, ClassVar
from fal_client import AsyncClient
from nodetool.metadata.types import AssetRef
from nodetool.workflows.base_node import ApiKeyMissingError, BaseNode
from nodetool.workflows.processing_context import ProcessingContext

logger = logging.getLogger(__name__)
DATA_URI_PATTERN = re.compile(r"data:([^;,]+)?;base64,[A-Za-z0-9+/=\r\n]+", re.IGNORECASE)
MAX_LOG_TEXT_LENGTH = 2000


class FALNode(BaseNode):
    """
    FAL Node for interacting with FAL AI services.
    Provides methods to submit and handle API requests to FAL endpoints.
    """

    _auto_save_asset: ClassVar[bool] = True
    _required_settings: ClassVar[list[str]] = ["FAL_API_KEY"]

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not FALNode

    async def get_client(self, context: ProcessingContext) -> AsyncClient:
        key = await context.get_secret("FAL_API_KEY")
        if key is None:
            raise ApiKeyMissingError("FAL_API_KEY is not set in the environment")

        return AsyncClient(key=key)

    def _sanitize_display_text(
        self, text: str, max_length: int = MAX_LOG_TEXT_LENGTH
    ) -> str:
        def _replace(match: re.Match[str]) -> str:
            mime = match.group(1) or "data"
            return f"[{mime} base64 omitted, {len(match.group(0))} chars]"

        sanitized = DATA_URI_PATTERN.sub(_replace, text)
        if len(sanitized) <= max_length:
            return sanitized

        truncated_chars = len(sanitized) - max_length
        return (
            f"{sanitized[:max_length]}... "
            f"(truncated {truncated_chars} chars)"
        )

    def _sanitize_log_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._sanitize_display_text(value)
        if isinstance(value, list):
            return [self._sanitize_log_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._sanitize_log_value(item) for item in value)
        if isinstance(value, dict):
            return {
                key: self._sanitize_log_value(item)
                for key, item in value.items()
            }
        return value

    async def _upload_asset_to_fal(
        self,
        client: AsyncClient,
        asset: AssetRef,
        context: ProcessingContext,
    ) -> str:
        """
        Upload an asset to FAL storage and return the URL.
        
        For assets that are already public HTTPS URLs, returns the URL directly.
        For assets with asset_id, data, or non-public URIs, uploads to FAL storage.
        
        Args:
            client: The FAL async client
            asset: The AssetRef to upload
            context: The processing context
            
        Returns:
            str: The URL of the asset in FAL storage or the original public URL
        """
        # If it's already a public HTTPS URL, use it directly
        if asset.uri and asset.uri.startswith("https://"):
            return asset.uri
        
        # For local URLs (http://, /api/storage/, file://, asset://), we need to upload to FAL
        # These are not accessible from FAL's remote servers
        
        # Get the asset bytes
        data_bytes = await context.asset_to_bytes(asset)
        
        # Determine content type based on asset type
        content_type_map = {
            "image": "image/png",
            "video": "video/mp4",
            "audio": "audio/wav",
        }
        content_type = content_type_map.get(asset.type, "application/octet-stream")
        
        # Upload to FAL storage using the client's upload method
        url = await client.upload(data_bytes, content_type)
        return url

    async def _convert_assets_recursive(
        self,
        client: AsyncClient,
        obj: Any,
        context: ProcessingContext,
    ) -> Any:
        """
        Recursively convert AssetRef objects to FAL storage URLs and Enum values to strings.
        
        Args:
            client: The FAL async client
            obj: Any object - scalar value, dict, list, AssetRef, or Enum
            context: The processing context
            
        Returns:
            The object with all AssetRef instances converted to URLs and Enums to strings
        """
        if isinstance(obj, AssetRef):
            return await self._upload_asset_to_fal(client, obj, context)
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                result[k] = await self._convert_assets_recursive(client, v, context)
            return result
        elif isinstance(obj, list):
            return [await self._convert_assets_recursive(client, item, context) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(await self._convert_assets_recursive(client, item, context) for item in obj)
        else:
            return obj

    async def submit_request(
        self,
        context: ProcessingContext,
        application: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Submit a request to a FAL AI endpoint and return the result.

        Args:
            application (str): The path to the FAL model (e.g., "fal-ai/flux/dev/image-to-image")
            arguments (Dict[str, Any]): The arguments to pass to the model
            with_logs (bool, optional): Whether to include logs in the response. Defaults to True.

        Returns:
            Dict[str, Any]: The result from the FAL API
        """
        client = await self.get_client(context)
        
        # Convert AssetRef objects to FAL storage URLs
        converted_arguments = await self._convert_assets_recursive(client, arguments, context)
        
        # Log the outgoing request parameters
        logger.info(f"FAL Request: {application}")
        logger.info(f"FAL Arguments: {self._sanitize_log_value(converted_arguments)}")

        try:
            handler = await client.submit(
                application,
                arguments=converted_arguments,
            )

            # Process events - only log useful ones
            async for event in handler.iter_events(with_logs=True):
                event_str = str(event)
                if "Queued" in event_str:
                    logger.debug(event)
                else:
                    logger.info(event)

            # Get the final result
            result = await handler.get()
            return result
        except Exception as exc:
            raise RuntimeError(self._sanitize_display_text(str(exc))) from exc
