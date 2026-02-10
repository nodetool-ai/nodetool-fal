from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import AudioRef
from nodetool.nodes.fal.types import Turn
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class ElevenLabsSpeechToText(FALNode):
    """
    ElevenLabs Speech to Text transcribes audio to text with high accuracy.
    audio, transcription, stt, elevenlabs, speech-to-text

    Use cases:
    - Transcribe audio files
    - Convert speech to text
    - Generate transcripts from audio
    - Extract text from recordings
    - Create captions from audio
    """

    language_code: str = Field(
        default="", description="Language code of the audio"
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to transcribe"
    )
    diarize: bool = Field(
        default=True, description="Whether to annotate who is speaking"
    )
    tag_audio_events: bool = Field(
        default=True, description="Tag audio events like laughter, applause, etc."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "language_code": self.language_code,
            "audio_url": self.audio,
            "diarize": self.diarize,
            "tag_audio_events": self.tag_audio_events,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/speech-to-text",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class ElevenLabsScribeV2(FALNode):
    """
    ElevenLabs Scribe V2 provides blazingly fast speech-to-text transcription.
    audio, transcription, stt, fast, elevenlabs, speech-to-text

    Use cases:
    - Fast audio transcription
    - Real-time speech recognition
    - Quick transcript generation
    - High-speed audio processing
    - Rapid speech-to-text conversion
    """

    language_code: str = Field(
        default="", description="Language code of the audio"
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to transcribe"
    )
    diarize: bool = Field(
        default=True, description="Whether to annotate who is speaking"
    )
    keyterms: list[str] = Field(
        default=[], description="Words or sentences to bias the model towards transcribing. Up to 100 keyterms, max 50 characters each. Adds 30% premium over base transcription price."
    )
    tag_audio_events: bool = Field(
        default=True, description="Tag audio events like laughter, applause, etc."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "language_code": self.language_code,
            "audio_url": self.audio,
            "diarize": self.diarize,
            "keyterms": self.keyterms,
            "tag_audio_events": self.tag_audio_events,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/speech-to-text/scribe-v2",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class SmartTurn(FALNode):
    """
    Pipecat's Smart Turn model provides native audio turn detection for conversations.
    audio, turn-detection, conversation, pipecat, speech-analysis

    Use cases:
    - Detect conversation turns
    - Identify speaker changes
    - Analyze dialogue timing
    - Detect speech boundaries
    - Process conversational audio
    """

    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file to be processed."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "audio_url": self.audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/smart-turn",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class SpeechToText(FALNode):
    """
    General-purpose speech-to-text model for accurate audio transcription.
    audio, transcription, stt, speech-to-text

    Use cases:
    - General audio transcription
    - Convert speech recordings to text
    - Generate audio transcripts
    - Process voice recordings
    - Extract text from speech
    """

    audio: AudioRef = Field(
        default=AudioRef(), description="Local filesystem path (or remote URL) to a long audio file"
    )
    use_pnc: bool = Field(
        default=True, description="Whether to use Canary's built-in punctuation & capitalization"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "audio_url": self.audio,
            "use_pnc": self.use_pnc,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/speech-to-text",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class SpeechToTextStream(FALNode):
    """
    Streaming speech-to-text for real-time audio transcription.
    audio, transcription, stt, streaming, real-time, speech-to-text

    Use cases:
    - Real-time transcription
    - Live audio captioning
    - Stream audio processing
    - Continuous speech recognition
    - Live speech-to-text conversion
    """

    audio: AudioRef = Field(
        default=AudioRef(), description="Local filesystem path (or remote URL) to a long audio file"
    )
    use_pnc: bool = Field(
        default=True, description="Whether to use Canary's built-in punctuation & capitalization"
    )

    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
            "audio_url": self.audio,
            "use_pnc": self.use_pnc,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/speech-to-text/stream",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio_stream"]

class SpeechToTextTurbo(FALNode):
    """
    High-speed speech-to-text model optimized for fast transcription.
    audio, transcription, stt, turbo, fast, speech-to-text

    Use cases:
    - Fast audio transcription
    - Quick speech recognition
    - Rapid transcript generation
    - High-speed processing
    - Efficient speech-to-text
    """

    audio: AudioRef = Field(
        default=AudioRef(), description="Local filesystem path (or remote URL) to a long audio file"
    )
    use_pnc: bool = Field(
        default=True, description="Whether to use Canary's built-in punctuation & capitalization"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "audio_url": self.audio,
            "use_pnc": self.use_pnc,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/speech-to-text/turbo",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class SpeechToTextTurboStream(FALNode):
    """
    High-speed streaming speech-to-text for real-time fast transcription.
    audio, transcription, stt, turbo, streaming, fast, speech-to-text

    Use cases:
    - Real-time fast transcription
    - Live fast captioning
    - High-speed streaming STT
    - Rapid live transcription
    - Efficient real-time processing
    """

    audio: AudioRef = Field(
        default=AudioRef(), description="Local filesystem path (or remote URL) to a long audio file"
    )
    use_pnc: bool = Field(
        default=True, description="Whether to use Canary's built-in punctuation & capitalization"
    )

    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
            "audio_url": self.audio,
            "use_pnc": self.use_pnc,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/speech-to-text/turbo/stream",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio_stream"]

class Whisper(FALNode):
    """
    OpenAI's Whisper model for robust multilingual speech recognition.
    audio, transcription, stt, whisper, multilingual, speech-to-text

    Use cases:
    - Multilingual transcription
    - Robust speech recognition
    - Transcribe multiple languages
    - Handle noisy audio
    - International audio processing
    """

    class Version(Enum):
        """
        Version of the model to use. All of the models are the Whisper large variant.
        """
        VALUE_3 = "3"

    class Language(Enum):
        """
        Language of the audio file. If set to null, the language will be
        automatically detected. Defaults to null.
        If translate is selected as the task, the audio will be translated to
        English, regardless of the language selected.
        """
        AF = "af"
        AM = "am"
        AR = "ar"
        AS = "as"
        AZ = "az"
        BA = "ba"
        BE = "be"
        BG = "bg"
        BN = "bn"
        BO = "bo"
        BR = "br"
        BS = "bs"
        CA = "ca"
        CS = "cs"
        CY = "cy"
        DA = "da"
        DE = "de"
        EL = "el"
        EN = "en"
        ES = "es"
        ET = "et"
        EU = "eu"
        FA = "fa"
        FI = "fi"
        FO = "fo"
        FR = "fr"
        GL = "gl"
        GU = "gu"
        HA = "ha"
        HAW = "haw"
        HE = "he"
        HI = "hi"
        HR = "hr"
        HT = "ht"
        HU = "hu"
        HY = "hy"
        ID = "id"
        IS = "is"
        IT = "it"
        JA = "ja"
        JW = "jw"
        KA = "ka"
        KK = "kk"
        KM = "km"
        KN = "kn"
        KO = "ko"
        LA = "la"
        LB = "lb"
        LN = "ln"
        LO = "lo"
        LT = "lt"
        LV = "lv"
        MG = "mg"
        MI = "mi"
        MK = "mk"
        ML = "ml"
        MN = "mn"
        MR = "mr"
        MS = "ms"
        MT = "mt"
        MY = "my"
        NE = "ne"
        NL = "nl"
        NN = "nn"
        NO = "no"
        OC = "oc"
        PA = "pa"
        PL = "pl"
        PS = "ps"
        PT = "pt"
        RO = "ro"
        RU = "ru"
        SA = "sa"
        SD = "sd"
        SI = "si"
        SK = "sk"
        SL = "sl"
        SN = "sn"
        SO = "so"
        SQ = "sq"
        SR = "sr"
        SU = "su"
        SV = "sv"
        SW = "sw"
        TA = "ta"
        TE = "te"
        TG = "tg"
        TH = "th"
        TK = "tk"
        TL = "tl"
        TR = "tr"
        TT = "tt"
        UK = "uk"
        UR = "ur"
        UZ = "uz"
        VI = "vi"
        YI = "yi"
        YO = "yo"
        ZH = "zh"

    class Task(Enum):
        """
        Task to perform on the audio file. Either transcribe or translate.
        """
        TRANSCRIBE = "transcribe"
        TRANSLATE = "translate"

    class ChunkLevel(Enum):
        """
        Level of the chunks to return. Either none, segment or word. `none` would imply that all of the audio will be transcribed without the timestamp tokens, we suggest to switch to `none` if you are not satisfied with the transcription quality, since it will usually improve the quality of the results. Switching to `none` will also provide minor speed ups in the transcription due to less amount of generated tokens. Notice that setting to none will produce **a single chunk with the whole transcription**.
        """
        NONE = "none"
        SEGMENT = "segment"
        WORD = "word"


    version: Version = Field(
        default=Version.VALUE_3, description="Version of the model to use. All of the models are the Whisper large variant."
    )
    batch_size: int = Field(
        default=64
    )
    language: Language | None = Field(
        default=None, description="Language of the audio file. If set to null, the language will be automatically detected. Defaults to null. If translate is selected as the task, the audio will be translated to English, regardless of the language selected."
    )
    prompt: str = Field(
        default="", description="Prompt to use for generation. Defaults to an empty string."
    )
    num_speakers: int = Field(
        default=0, description="Number of speakers in the audio file. Defaults to null. If not provided, the number of speakers will be automatically detected."
    )
    task: Task = Field(
        default=Task.TRANSCRIBE, description="Task to perform on the audio file. Either transcribe or translate."
    )
    chunk_level: ChunkLevel = Field(
        default=ChunkLevel.SEGMENT, description="Level of the chunks to return. Either none, segment or word. `none` would imply that all of the audio will be transcribed without the timestamp tokens, we suggest to switch to `none` if you are not satisfied with the transcription quality, since it will usually improve the quality of the results. Switching to `none` will also provide minor speed ups in the transcription due to less amount of generated tokens. Notice that setting to none will produce **a single chunk with the whole transcription**."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to transcribe. Supported formats: mp3, mp4, mpeg, mpga, m4a, wav or webm."
    )
    diarize: bool = Field(
        default=False, description="Whether to diarize the audio file. Defaults to false. Setting to true will add costs proportional to diarization inference time."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "version": self.version.value,
            "batch_size": self.batch_size,
            "language": self.language.value if self.language else None,
            "prompt": self.prompt,
            "num_speakers": self.num_speakers,
            "task": self.task.value,
            "chunk_level": self.chunk_level.value,
            "audio_url": self.audio,
            "diarize": self.diarize,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/whisper",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class Wizper(FALNode):
    """
    Wizper provides fast and accurate speech-to-text transcription.
    audio, transcription, stt, wizper, fast, speech-to-text

    Use cases:
    - Fast accurate transcription
    - Quick speech recognition
    - Efficient audio processing
    - Rapid text extraction
    - Speedy speech-to-text
    """

    class Task(Enum):
        """
        Task to perform on the audio file. Either transcribe or translate.
        """
        TRANSCRIBE = "transcribe"
        TRANSLATE = "translate"


    language: str = Field(
        default="en", description="Language of the audio file. If translate is selected as the task, the audio will be translated to English, regardless of the language selected. If `None` is passed, the language will be automatically detected. This will also increase the inference time."
    )
    version: str = Field(
        default="3", description="Version of the model to use. All of the models are the Whisper large variant."
    )
    max_segment_len: int = Field(
        default=29, description="Maximum speech segment duration in seconds before splitting."
    )
    task: Task = Field(
        default=Task.TRANSCRIBE, description="Task to perform on the audio file. Either transcribe or translate."
    )
    chunk_level: str = Field(
        default="segment", description="Level of the chunks to return."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to transcribe. Supported formats: mp3, mp4, mpeg, mpga, m4a, wav or webm."
    )
    merge_chunks: bool = Field(
        default=True, description="Whether to merge consecutive chunks. When enabled, chunks are merged if their combined duration does not exceed max_segment_len."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "language": self.language,
            "version": self.version,
            "max_segment_len": self.max_segment_len,
            "task": self.task.value,
            "chunk_level": self.chunk_level,
            "audio_url": self.audio,
            "merge_chunks": self.merge_chunks,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wizper",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]