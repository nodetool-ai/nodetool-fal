import logging
from typing import Any, ClassVar
from fal_client import AsyncClient
from nodetool.workflows.base_node import ApiKeyMissingError, BaseNode
from nodetool.workflows.processing_context import ProcessingContext

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
