from pydantic import Field

from nodetool.metadata.types import VideoRef, ImageRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class VideoUpscaler(FALNode):
    """
    Video Upscaler enhances video resolution using AI.
    video, upscaling, enhancement, super-resolution

    Use cases:
    - Upscale low-resolution videos
    - Enhance video quality
    - Improve video for larger displays
    - Restore old videos
    - Prepare videos for high-res output
    """

    video: VideoRef = Field(default=VideoRef(), description="The video to upscale")
    scale: float = Field(default=2.0, ge=1.0, le=8.0, description="Upscaling factor (1-8)")

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_bytes = await context.asset_to_bytes(self.video)
        video_url = await client.upload(video_bytes, "video/mp4")

        arguments = {
            "video_url": video_url,
            "scale": self.scale,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/video-upscaler",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "scale"]


class RIFE(FALNode):
    """
    RIFE (Real-time Intermediate Flow Estimation) interpolates frames for smooth video.
    video, interpolation, frame-rate, smoothing

    Use cases:
    - Increase video frame rate
    - Create smooth slow motion
    - Improve video fluidity
    - Generate intermediate frames
    - Enhance animation smoothness
    """

    start_image: ImageRef = Field(default=ImageRef(), description="The start frame")
    end_image: ImageRef = Field(default=ImageRef(), description="The end frame")
    num_frames: int = Field(
        default=2, ge=1, le=16, description="Number of intermediate frames"
    )

    async def process(self, context: ProcessingContext) -> list:
        start_base64 = await context.image_to_base64(self.start_image)
        end_base64 = await context.image_to_base64(self.end_image)

        arguments = {
            "start_image_url": f"data:image/png;base64,{start_base64}",
            "end_image_url": f"data:image/png;base64,{end_base64}",
            "num_frames": self.num_frames,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/rife",
            arguments=arguments,
        )
        return [ImageRef(uri=img["url"]) for img in res.get("images", [])]

    @classmethod
    def get_basic_fields(cls):
        return ["start_image", "end_image", "num_frames"]


class RIFEVideo(FALNode):
    """
    RIFE Video interpolates video frames for increased frame rate.
    video, interpolation, frame-rate, enhancement

    Use cases:
    - Double video frame rate
    - Create slow motion videos
    - Improve video smoothness
    - Enhance low-fps footage
    - Generate high-fps content
    """

    video: VideoRef = Field(default=VideoRef(), description="The video to interpolate")
    multiplier: int = Field(default=2, ge=2, le=4, description="Frame rate multiplier")

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_bytes = await context.asset_to_bytes(self.video)
        video_url = await client.upload(video_bytes, "video/mp4")

        arguments = {
            "video_url": video_url,
            "multiplier": self.multiplier,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/rife/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "multiplier"]


class SyncLipsyncV2(FALNode):
    """
    Sync Lipsync V2 synchronizes lip movements to audio.
    video, lipsync, audio, synchronization

    Use cases:
    - Sync lips to new audio
    - Create talking head videos
    - Dub videos in other languages
    - Generate speaking animations
    - Create video voice-overs
    """

    video: VideoRef = Field(default=VideoRef(), description="The video with the face")
    audio: str = Field(default="", description="URL to the audio file for lipsync")

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_bytes = await context.asset_to_bytes(self.video)
        video_url = await client.upload(video_bytes, "video/mp4")

        arguments = {
            "video_url": video_url,
            "audio_url": self.audio,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/sync-lipsync/v2",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "audio"]


class LiveAvatar(FALNode):
    """
    Live Avatar creates animated avatars from images and audio.
    video, avatar, animation, audio-driven

    Use cases:
    - Create talking avatars
    - Generate animated presentations
    - Produce video content from photos
    - Create virtual presenters
    - Generate video messages
    """

    image: ImageRef = Field(default=ImageRef(), description="The avatar image")
    audio: str = Field(default="", description="URL to the driving audio")

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "audio_url": self.audio,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/live-avatar",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "audio"]


class TopazVideoUpscale(FALNode):
    """
    Topaz Video Upscale enhances video quality using advanced AI.
    video, upscaling, enhancement, topaz, professional

    Use cases:
    - Professional video upscaling
    - Restore archival footage
    - Enhance video for broadcast
    - Improve video quality
    - Prepare videos for 4K display
    """

    video: VideoRef = Field(default=VideoRef(), description="The video to upscale")
    scale: int = Field(default=2, ge=2, le=4, description="Upscaling factor")

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_bytes = await context.asset_to_bytes(self.video)
        video_url = await client.upload(video_bytes, "video/mp4")

        arguments = {
            "video_url": video_url,
            "scale": self.scale,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/topaz/upscale/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "scale"]


class FaceSwapVideo(FALNode):
    """
    Swap faces in videos using a source face image. Replaces faces in the target video with the source face while maintaining natural motion and expressions.
    face-swap, video-editing, face-replacement, deep-fake, video-manipulation

    Use cases:
    - Create face-swapped video content
    - Generate creative video edits
    - Produce entertainment content
    - Test different faces in video footage
    - Create video memes and parodies
    """

    source_face: ImageRef = Field(
        default=ImageRef(), description="Source face image to swap into video"
    )
    target_video: VideoRef = Field(
        default=VideoRef(), description="Target video to swap face in (max 25 minutes)"
    )
    enable_occlusion_prevention: bool = Field(
        default=False,
        description="Enable occlusion prevention for faces covered by hands/objects (costs 2x more)",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        
        source_base64 = await context.image_to_base64(self.source_face)
        video_bytes = await context.asset_to_bytes(self.target_video)
        video_url = await client.upload(video_bytes, "video/mp4")

        arguments = {
            "source_face_url": f"data:image/png;base64,{source_base64}",
            "target_video_url": video_url,
        }

        if self.enable_occlusion_prevention:
            arguments["enable_occlusion_prevention"] = self.enable_occlusion_prevention

        res = await self.submit_request(
            context=context,
            application="half-moon-ai/ai-face-swap/faceswapvideo",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["source_face", "target_video"]
