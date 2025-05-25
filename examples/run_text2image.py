import asyncio
import os

from nodetool.nodes.fal.text_to_image import IdeogramV2
from nodetool.workflows.processing_context import ProcessingContext


async def main() -> None:
    # Create node with a simple prompt
    node = IdeogramV2(prompt="a colorful parrot")

    # FAL_API_KEY must be provided via environment variables
    context = ProcessingContext(
        environment={"FAL_API_KEY": os.getenv("FAL_API_KEY", "")}
    )

    image = await node.process(context)
    print("Generated image URL:", image.uri)


if __name__ == "__main__":
    asyncio.run(main())
