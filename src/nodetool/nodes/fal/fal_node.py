import logging
from typing import Any, ClassVar
from fal_client import AsyncClient
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

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not FALNode

    async def get_client(self, context: ProcessingContext) -> AsyncClient:
        key = await context.get_secret("FAL_API_KEY")
        if key is None:
            raise ApiKeyMissingError("FAL_API_KEY is not set in the environment")

        return AsyncClient(key=key)

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
        handler = await client.submit(
            application,
            arguments=arguments,
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

        # Count video outputs and estimate video seconds
        video_seconds = 0.0
        if "video" in result:
            # For video outputs, we don't have duration in the response
            # Could be estimated from model parameters if available
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
