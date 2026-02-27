import logging
from enum import Enum
from typing import Any, ClassVar
from fal_client import AsyncClient
from nodetool.metadata.types import AssetRef
from nodetool.workflows.base_node import ApiKeyMissingError, BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.providers.cost_calculator import UsageInfo

logger = logging.getLogger(__name__)


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
        logger.info(f"FAL Arguments: {converted_arguments}")
        
        handler = await client.submit(
            application,
            arguments=converted_arguments,
        )

        # Track inference time from events
        inference_time = 0.0

        # Process events - only log useful ones
        async for event in handler.iter_events(with_logs=True):
            event_str = str(event)
            if "Queued" in event_str:
                logger.debug(event)
            else:
                logger.info(event)

            # Extract metrics from completed events
            if hasattr(event, "metrics") and event.metrics:
                if "inference_time" in event.metrics:
                    inference_time = event.metrics["inference_time"]

        # Get the final result
        result = await handler.get()

        # Track cost in context
        self._track_cost(context, application, result, inference_time)

        return result

    def _track_cost(
        self,
        context: ProcessingContext,
        application: str,
        result: dict[str, Any],
        inference_time: float,
    ) -> None:
        """
        Track the cost of a FAL API operation.

        Args:
            context: The processing context
            application: The FAL model path
            result: The result from the FAL API
            inference_time: The inference time in seconds
        """
        # Count output images
        image_count = 0
        if "images" in result and isinstance(result["images"], list):
            image_count = len(result["images"])
        elif "image" in result:
            image_count = 1

        # Video duration is not available in FAL API responses
        # This could be estimated from model parameters in the future
        video_seconds = 0.0

        # Create usage info
        usage_info = UsageInfo(
            duration_seconds=inference_time,
            image_count=image_count,
            video_seconds=video_seconds,
        )

        # Track the operation cost
        context.track_operation_cost(
            model=application,
            provider="fal",
            usage_info=usage_info,
            node_id=self.id,
            operation_type="prediction",
        )
