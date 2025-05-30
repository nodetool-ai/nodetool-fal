from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.fal.speech_to_text
import nodetool.nodes.fal.speech_to_text
import nodetool.nodes.fal.speech_to_text


class Whisper(GraphNode):
    """
    Whisper is a model for speech transcription and translation that can transcribe audio in multiple languages and optionally translate to English.
    speech, audio, transcription, translation, transcribe, translate, multilingual, speech-to-text, audio-to-text

    Use cases:
    - Transcribe spoken content to text
    - Translate speech to English
    - Generate subtitles and captions
    - Create text records of audio content
    - Analyze multilingual audio content
    """

    TaskEnum: typing.ClassVar[type] = nodetool.nodes.fal.speech_to_text.TaskEnum
    LanguageEnum: typing.ClassVar[type] = nodetool.nodes.fal.speech_to_text.LanguageEnum
    ChunkLevelEnum: typing.ClassVar[type] = (
        nodetool.nodes.fal.speech_to_text.ChunkLevelEnum
    )
    audio: types.AudioRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.AudioRef(type="audio", uri="", asset_id=None, data=None),
        description="The audio file to transcribe",
    )
    task: nodetool.nodes.fal.speech_to_text.TaskEnum = Field(
        default=nodetool.nodes.fal.speech_to_text.TaskEnum.TRANSCRIBE,
        description="Task to perform on the audio file",
    )
    language: nodetool.nodes.fal.speech_to_text.LanguageEnum = Field(
        default=nodetool.nodes.fal.speech_to_text.LanguageEnum.EN,
        description="Language of the audio file. If not set, will be auto-detected",
    )
    diarize: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Whether to perform speaker diarization"
    )
    chunk_level: nodetool.nodes.fal.speech_to_text.ChunkLevelEnum = Field(
        default=nodetool.nodes.fal.speech_to_text.ChunkLevelEnum.SEGMENT,
        description="Level of detail for timestamp chunks",
    )
    num_speakers: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1,
        description="Number of speakers in the audio. If not set, will be auto-detected",
    )
    batch_size: int | GraphNode | tuple[GraphNode, str] = Field(
        default=64, description="Batch size for processing"
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Optional prompt to guide the transcription"
    )

    @classmethod
    def get_node_type(cls):
        return "fal.speech_to_text.Whisper"
