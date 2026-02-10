from enum import Enum
from pydantic import Field
from nodetool.metadata.types import AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class ResembleAiChatterboxhdSpeechToSpeech(FALNode):
    """
    Transform voices using Resemble AI's Chatterbox. Convert audio to new voices or your own samples, with expressive results and built-in perceptual watermarking.
    speech, voice, transformation, cloning

    Use cases:
    - Voice cloning and transformation
    - Real-time voice conversion
    - Voice style transfer
    - Speech enhancement
    - Accent conversion
    """

    class TargetVoice(Enum):
        """
        The voice to use for the speech-to-speech request. If neither target_voice nor target_voice_audio_url are provided, a random target voice will be used.
        """
        AURORA = "Aurora"
        BLADE = "Blade"
        BRITNEY = "Britney"
        CARL = "Carl"
        CLIFF = "Cliff"
        RICHARD = "Richard"
        RICO = "Rico"
        SIOBHAN = "Siobhan"
        VICKY = "Vicky"


    high_quality_audio: bool = Field(
        default=False, description="If True, the generated audio will be upscaled to 48kHz. The generation of the audio will take longer, but the quality will be higher. If False, the generated audio will be 24kHz."
    )
    target_voice_audio: AudioRef = Field(
        default=AudioRef(), description="URL to the audio file which represents the voice of the output audio. If provided, this will override the target_voice setting. If neither target_voice nor target_voice_audio_url are provided, the default target voice will be used."
    )
    source_audio: AudioRef = Field(
        default=AudioRef(), description="URL to the source audio file to be voice-converted."
    )
    target_voice: TargetVoice | None = Field(
        default=None, description="The voice to use for the speech-to-speech request. If neither target_voice nor target_voice_audio_url are provided, a random target voice will be used."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "high_quality_audio": self.high_quality_audio,
            "target_voice_audio_url": self.target_voice_audio,
            "source_audio_url": self.source_audio,
            "target_voice": self.target_voice.value if self.target_voice else None,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="resemble-ai/chatterboxhd/speech-to-speech",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["high_quality_audio", "target_voice_audio", "source_audio", "target_voice"]

class ChatterboxSpeechToSpeech(FALNode):
    """
    Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. Use the first tts from resemble ai.
    speech, voice, transformation, cloning

    Use cases:
    - Voice cloning and transformation
    - Real-time voice conversion
    - Voice style transfer
    - Speech enhancement
    - Accent conversion
    """

    source_audio: AudioRef = Field(
        default=AudioRef()
    )
    target_voice_audio: AudioRef = Field(
        default=AudioRef(), description="Optional URL to an audio file to use as a reference for the generated speech. If provided, the model will try to match the style and tone of the reference audio."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "source_audio_url": self.source_audio,
            "target_voice_audio_url": self.target_voice_audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/chatterbox/speech-to-speech",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["source_audio", "target_voice_audio"]