from pydantic import Field

from nodetool.metadata.types import ImageRef, VideoRef, AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class Florence2Caption(FALNode):
    """
    Florence 2 Caption generates detailed captions for images.
    vision, caption, understanding, florence, image-to-text

    Use cases:
    - Generate image descriptions
    - Create alt text for images
    - Analyze image content
    - Produce accessibility content
    - Create image metadata
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to caption")

    async def process(self, context: ProcessingContext) -> str:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/florence-2-large/caption",
            arguments=arguments,
        )
        return res.get("caption", "")

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class Florence2DetailedCaption(FALNode):
    """
    Florence 2 Detailed Caption generates comprehensive image descriptions.
    vision, caption, understanding, florence, detailed

    Use cases:
    - Generate detailed image descriptions
    - Create comprehensive alt text
    - Analyze complex images
    - Produce rich metadata
    - Create detailed content descriptions
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to caption")

    async def process(self, context: ProcessingContext) -> str:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/florence-2-large/detailed-caption",
            arguments=arguments,
        )
        return res.get("caption", "")

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class Florence2ObjectDetection(FALNode):
    """
    Florence 2 Object Detection identifies and locates objects in images.
    vision, object-detection, understanding, florence

    Use cases:
    - Detect objects in images
    - Identify items in photos
    - Analyze image content
    - Create object inventories
    - Enable visual search
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to analyze")

    async def process(self, context: ProcessingContext) -> dict:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/florence-2-large/object-detection",
            arguments=arguments,
        )
        return {
            "objects": res.get("objects", []),
            "labels": res.get("labels", []),
        }

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

    @classmethod
    def return_type(cls):
        return {"objects": list, "labels": list}


class Florence2OCR(FALNode):
    """
    Florence 2 OCR extracts text from images.
    vision, ocr, text-extraction, florence

    Use cases:
    - Extract text from images
    - Read documents
    - Process screenshots
    - Digitize printed text
    - Extract labels and signs
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to extract text from"
    )

    async def process(self, context: ProcessingContext) -> str:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/florence-2-large/ocr",
            arguments=arguments,
        )
        return res.get("text", "")

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class Moondream2(FALNode):
    """
    Moondream2 is a small but capable vision-language model for image understanding.
    vision, vlm, understanding, moondream, image-to-text

    Use cases:
    - Answer questions about images
    - Analyze image content
    - Generate descriptions
    - Visual question answering
    - Image understanding tasks
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to analyze")
    prompt: str = Field(
        default="Describe this image.",
        description="The question or prompt about the image",
    )

    async def process(self, context: ProcessingContext) -> str:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/moondream2",
            arguments=arguments,
        )
        return res.get("output", "")

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class LlavaNext(FALNode):
    """
    LLaVA-NeXT is an advanced vision-language model for image understanding.
    vision, vlm, understanding, llava, multimodal

    Use cases:
    - Complex image analysis
    - Visual question answering
    - Image captioning
    - Scene understanding
    - Multi-turn visual conversations
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to analyze")
    prompt: str = Field(
        default="Describe this image in detail.",
        description="The question or prompt about the image",
    )

    async def process(self, context: ProcessingContext) -> str:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/llava-next",
            arguments=arguments,
        )
        return res.get("output", "")

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class VideoUnderstanding(FALNode):
    """
    Video Understanding analyzes video content and generates descriptions.
    vision, video, understanding, analysis

    Use cases:
    - Analyze video content
    - Generate video summaries
    - Extract video descriptions
    - Understand video scenes
    - Create video metadata
    """

    video: VideoRef = Field(default=VideoRef(), description="The video to analyze")
    prompt: str = Field(
        default="Describe what happens in this video.",
        description="The question or prompt about the video",
    )

    async def process(self, context: ProcessingContext) -> str:
        client = await self.get_client(context)
        video_bytes = await context.asset_to_bytes(self.video)
        video_url = await client.upload(video_bytes, "video/mp4")

        arguments = {
            "video_url": video_url,
            "prompt": self.prompt,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/video-understanding",
            arguments=arguments,
        )
        return res.get("output", "")

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]


class AudioUnderstanding(FALNode):
    """
    Audio Understanding analyzes audio content and generates descriptions.
    audio, understanding, analysis

    Use cases:
    - Analyze audio content
    - Generate audio descriptions
    - Understand sound scenes
    - Create audio metadata
    - Identify audio events
    """

    audio: AudioRef = Field(default=AudioRef(), description="The audio to analyze")
    prompt: str = Field(
        default="Describe what you hear in this audio.",
        description="The question or prompt about the audio",
    )

    async def process(self, context: ProcessingContext) -> str:
        client = await self.get_client(context)
        audio_bytes = await context.asset_to_bytes(self.audio)
        audio_url = await client.upload(audio_bytes, "audio/mp3")

        arguments = {
            "audio_url": audio_url,
            "prompt": self.prompt,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/audio-understanding",
            arguments=arguments,
        )
        return res.get("output", "")

    @classmethod
    def get_basic_fields(cls):
        return ["audio", "prompt"]


class GotOCR(FALNode):
    """
    GOT-OCR V2 is an advanced OCR model for extracting text from images.
    vision, ocr, text-extraction, got

    Use cases:
    - Extract text from complex images
    - Read handwritten text
    - Process documents
    - Digitize printed material
    - Extract multilingual text
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to extract text from"
    )

    async def process(self, context: ProcessingContext) -> str:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/got-ocr/v2",
            arguments=arguments,
        )
        return res.get("text", "")

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class VideoPromptGenerator(FALNode):
    """
    Video Prompt Generator creates detailed prompts for video generation from reference videos.
    vision, video, prompt, generation

    Use cases:
    - Generate prompts from reference videos
    - Create video descriptions
    - Extract video styles
    - Analyze video content for prompts
    - Enable video-to-video workflows
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The reference video to analyze"
    )

    async def process(self, context: ProcessingContext) -> str:
        client = await self.get_client(context)
        video_bytes = await context.asset_to_bytes(self.video)
        video_url = await client.upload(video_bytes, "video/mp4")

        arguments = {
            "video_url": video_url,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/video-prompt-generator",
            arguments=arguments,
        )
        return res.get("prompt", "")

    @classmethod
    def get_basic_fields(cls):
        return ["video"]
