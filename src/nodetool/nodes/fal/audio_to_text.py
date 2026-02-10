from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import AudioRef
from nodetool.nodes.fal.types import Speaker
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class NemotronAsrStream(FALNode):
    """
    Use the fast speed and pin point accuracy of nemotron to transcribe your texts.
    speech, recognition, transcription, audio-analysis

    Use cases:
    - Speech recognition
    - Audio transcription
    - Speaker diarization
    - Voice activity detection
    - Meeting transcription
    """

    class Acceleration(Enum):
        """
        Controls the speed/accuracy trade-off. 'none' = best accuracy (1.12s chunks, ~7.16% WER), 'low' = balanced (0.56s chunks, ~7.22% WER), 'medium' = faster (0.16s chunks, ~7.84% WER), 'high' = fastest (0.08s chunks, ~8.53% WER).
        """
        NONE = "none"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"


    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="Controls the speed/accuracy trade-off. 'none' = best accuracy (1.12s chunks, ~7.16% WER), 'low' = balanced (0.56s chunks, ~7.22% WER), 'medium' = faster (0.16s chunks, ~7.84% WER), 'high' = fastest (0.08s chunks, ~8.53% WER)."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file."
    )

    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
            "acceleration": self.acceleration.value,
            "audio_url": self.audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/nemotron/asr/stream",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["acceleration", "audio"]

class NemotronAsr(FALNode):
    """
    Use the fast speed and pin point accuracy of nemotron to transcribe your texts.
    speech, recognition, transcription, audio-analysis

    Use cases:
    - Speech recognition
    - Audio transcription
    - Speaker diarization
    - Voice activity detection
    - Meeting transcription
    """

    class Acceleration(Enum):
        """
        Controls the speed/accuracy trade-off. 'none' = best accuracy (1.12s chunks, ~7.16% WER), 'low' = balanced (0.56s chunks, ~7.22% WER), 'medium' = faster (0.16s chunks, ~7.84% WER), 'high' = fastest (0.08s chunks, ~8.53% WER).
        """
        NONE = "none"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"


    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="Controls the speed/accuracy trade-off. 'none' = best accuracy (1.12s chunks, ~7.16% WER), 'low' = balanced (0.56s chunks, ~7.22% WER), 'medium' = faster (0.16s chunks, ~7.84% WER), 'high' = fastest (0.08s chunks, ~8.53% WER)."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "acceleration": self.acceleration.value,
            "audio_url": self.audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/nemotron/asr",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["acceleration", "audio"]

class SileroVad(FALNode):
    """
    Detect speech presence and timestamps with accuracy and speed using the ultra-lightweight Silero VAD model
    speech, recognition, transcription, audio-analysis

    Use cases:
    - Speech recognition
    - Audio transcription
    - Speaker diarization
    - Voice activity detection
    - Meeting transcription
    """

    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio to get speech timestamps from."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "audio_url": self.audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/silero-vad",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]