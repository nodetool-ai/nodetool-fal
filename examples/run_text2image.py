import asyncio
import logging
import os

from nodetool.nodes.fal.text_to_image import IdeogramV2
from nodetool.workflows.processing_context import ProcessingContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    # Create node with a simple prompt
    node = IdeogramV2(prompt="a colorful parrot")

    # FAL_API_KEY must be provided via environment variables
    context = ProcessingContext(
        environment={"FAL_API_KEY": os.getenv("FAL_API_KEY", "")}
    )

    image = await node.process(context)
    logger.info("Generated image URL: %s", image.uri)

    # Download the generated image so it can be uploaded as a workflow artifact
    import urllib.request

    with (
        urllib.request.urlopen(image.uri) as response,
        open("generated_image.png", "wb") as out_file,
    ):
        out_file.write(response.read())
    logger.info("Image saved to generated_image.png")


if __name__ == "__main__":
    asyncio.run(main())
