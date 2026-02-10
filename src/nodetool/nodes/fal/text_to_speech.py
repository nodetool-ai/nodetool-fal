from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.nodes.fal.types import VibeVoiceSpeaker  # noqa: F401
from nodetool.workflows.processing_context import ProcessingContext


class Qwen3TtsTextToSpeech17B(FALNode):
    """
    Qwen-3 TTS 1.7B generates natural-sounding speech from text using the large 1.7-billion parameter model.
    audio, tts, qwen, 1.7b, text-to-speech, speech-synthesis

    Use cases:
    - Generate natural-sounding speech from text
    - Create voice-overs for videos
    - Produce audiobook narration
    - Generate spoken content for applications
    - Create text-to-speech for accessibility
    """

    class Voice(Enum):
        """
        The voice to be used for speech synthesis, will be ignored if a speaker embedding is provided. Check out the **[documentation](https://github.com/QwenLM/Qwen3-TTS/tree/main?tab=readme-ov-file#custom-voice-generate)** for each voice's details and which language they primarily support.
        """
        VIVIAN = "Vivian"
        SERENA = "Serena"
        UNCLE_FU = "Uncle_Fu"
        DYLAN = "Dylan"
        ERIC = "Eric"
        RYAN = "Ryan"
        AIDEN = "Aiden"
        ONO_ANNA = "Ono_Anna"
        SOHEE = "Sohee"

    class Language(Enum):
        """
        The language of the voice.
        """
        AUTO = "Auto"
        ENGLISH = "English"
        CHINESE = "Chinese"
        SPANISH = "Spanish"
        FRENCH = "French"
        GERMAN = "German"
        ITALIAN = "Italian"
        JAPANESE = "Japanese"
        KOREAN = "Korean"
        PORTUGUESE = "Portuguese"
        RUSSIAN = "Russian"


    prompt: str = Field(
        default="", description="Optional prompt to guide the style of the generated speech. This prompt will be ignored if a speaker embedding is provided."
    )
    speaker_voice_embedding_file_url: str = Field(
        default="", description="URL to a speaker embedding file in safetensors format, from `fal-ai/qwen-3-tts/clone-voice` endpoint. If provided, the TTS model will use the cloned voice for synthesis instead of the predefined voices."
    )
    top_p: float = Field(
        default=1, description="Top-p sampling parameter."
    )
    repetition_penalty: float = Field(
        default=1.05, description="Penalty to reduce repeated tokens/codes."
    )
    subtalker_temperature: float = Field(
        default=0.9, description="Temperature for sub-talker sampling."
    )
    top_k: int = Field(
        default=50, description="Top-k sampling parameter."
    )
    voice: Voice | None = Field(
        default=None, description="The voice to be used for speech synthesis, will be ignored if a speaker embedding is provided. Check out the **[documentation](https://github.com/QwenLM/Qwen3-TTS/tree/main?tab=readme-ov-file#custom-voice-generate)** for each voice's details and which language they primarily support."
    )
    reference_text: str = Field(
        default="", description="Optional reference text that was used when creating the speaker embedding. Providing this can improve synthesis quality when using a cloned voice."
    )
    temperature: float = Field(
        default=0.9, description="Sampling temperature; higher => more random."
    )
    language: Language = Field(
        default=Language.AUTO, description="The language of the voice."
    )
    subtalker_top_k: int = Field(
        default=50, description="Top-k for sub-talker sampling."
    )
    text: str = Field(
        default="", description="The text to be converted to speech."
    )
    max_new_tokens: int = Field(
        default=200, description="Maximum number of new codec tokens to generate."
    )
    subtalker_dosample: bool = Field(
        default=True, description="Sampling switch for the sub-talker."
    )
    subtalker_top_p: float = Field(
        default=1, description="Top-p for sub-talker sampling."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "speaker_voice_embedding_file_url": self.speaker_voice_embedding_file_url,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "subtalker_temperature": self.subtalker_temperature,
            "top_k": self.top_k,
            "voice": self.voice.value if self.voice else None,
            "reference_text": self.reference_text,
            "temperature": self.temperature,
            "language": self.language.value,
            "subtalker_top_k": self.subtalker_top_k,
            "text": self.text,
            "max_new_tokens": self.max_new_tokens,
            "subtalker_dosample": self.subtalker_dosample,
            "subtalker_top_p": self.subtalker_top_p,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-3-tts/text-to-speech/1.7b",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class Qwen3TtsTextToSpeech06B(FALNode):
    """
    Qwen-3 TTS 0.6B generates speech from text efficiently using the compact 600-million parameter model.
    audio, tts, qwen, 0.6b, efficient, text-to-speech

    Use cases:
    - Generate speech efficiently from text
    - Create fast voice-overs
    - Produce quick audio narration
    - Generate spoken content with low latency
    - Create efficient text-to-speech
    """

    class Voice(Enum):
        """
        The voice to be used for speech synthesis, will be ignored if a speaker embedding is provided. Check out the **[documentation](https://github.com/QwenLM/Qwen3-TTS/tree/main?tab=readme-ov-file#custom-voice-generate)** for each voice's details and which language they primarily support.
        """
        VIVIAN = "Vivian"
        SERENA = "Serena"
        UNCLE_FU = "Uncle_Fu"
        DYLAN = "Dylan"
        ERIC = "Eric"
        RYAN = "Ryan"
        AIDEN = "Aiden"
        ONO_ANNA = "Ono_Anna"
        SOHEE = "Sohee"

    class Language(Enum):
        """
        The language of the voice.
        """
        AUTO = "Auto"
        ENGLISH = "English"
        CHINESE = "Chinese"
        SPANISH = "Spanish"
        FRENCH = "French"
        GERMAN = "German"
        ITALIAN = "Italian"
        JAPANESE = "Japanese"
        KOREAN = "Korean"
        PORTUGUESE = "Portuguese"
        RUSSIAN = "Russian"


    prompt: str = Field(
        default="", description="Optional prompt to guide the style of the generated speech. This prompt will be ignored if a speaker embedding is provided."
    )
    speaker_voice_embedding_file_url: str = Field(
        default="", description="URL to a speaker embedding file in safetensors format, from `fal-ai/qwen-3-tts/clone-voice/0.6b` endpoint. If provided, the TTS model will use the cloned voice for synthesis instead of the predefined voices."
    )
    top_p: float = Field(
        default=1, description="Top-p sampling parameter."
    )
    repetition_penalty: float = Field(
        default=1.05, description="Penalty to reduce repeated tokens/codes."
    )
    subtalker_temperature: float = Field(
        default=0.9, description="Temperature for sub-talker sampling."
    )
    top_k: int = Field(
        default=50, description="Top-k sampling parameter."
    )
    voice: Voice | None = Field(
        default=None, description="The voice to be used for speech synthesis, will be ignored if a speaker embedding is provided. Check out the **[documentation](https://github.com/QwenLM/Qwen3-TTS/tree/main?tab=readme-ov-file#custom-voice-generate)** for each voice's details and which language they primarily support."
    )
    reference_text: str = Field(
        default="", description="Optional reference text that was used when creating the speaker embedding. Providing this can improve synthesis quality when using a cloned voice."
    )
    temperature: float = Field(
        default=0.9, description="Sampling temperature; higher => more random."
    )
    language: Language = Field(
        default=Language.AUTO, description="The language of the voice."
    )
    subtalker_top_k: int = Field(
        default=50, description="Top-k for sub-talker sampling."
    )
    text: str = Field(
        default="", description="The text to be converted to speech."
    )
    max_new_tokens: int = Field(
        default=200, description="Maximum number of new codec tokens to generate."
    )
    subtalker_dosample: bool = Field(
        default=True, description="Sampling switch for the sub-talker."
    )
    subtalker_top_p: float = Field(
        default=1, description="Top-p for sub-talker sampling."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "speaker_voice_embedding_file_url": self.speaker_voice_embedding_file_url,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "subtalker_temperature": self.subtalker_temperature,
            "top_k": self.top_k,
            "voice": self.voice.value if self.voice else None,
            "reference_text": self.reference_text,
            "temperature": self.temperature,
            "language": self.language.value,
            "subtalker_top_k": self.subtalker_top_k,
            "text": self.text,
            "max_new_tokens": self.max_new_tokens,
            "subtalker_dosample": self.subtalker_dosample,
            "subtalker_top_p": self.subtalker_top_p,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-3-tts/text-to-speech/0.6b",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class Qwen3TtsVoiceDesign17B(FALNode):
    """
    Qwen-3 TTS Voice Design 1.7B creates custom voice characteristics for personalized speech synthesis.
    audio, tts, qwen, voice-design, custom, 1.7b

    Use cases:
    - Design custom voice characteristics
    - Create personalized speech synthesis
    - Generate unique voice styles
    - Produce custom voice-overs
    - Create tailored speech synthesis
    """

    class Language(Enum):
        """
        The language of the voice to be designed.
        """
        AUTO = "Auto"
        ENGLISH = "English"
        CHINESE = "Chinese"
        SPANISH = "Spanish"
        FRENCH = "French"
        GERMAN = "German"
        ITALIAN = "Italian"
        JAPANESE = "Japanese"
        KOREAN = "Korean"
        PORTUGUESE = "Portuguese"
        RUSSIAN = "Russian"


    repetition_penalty: float = Field(
        default=1.05, description="Penalty to reduce repeated tokens/codes."
    )
    subtalker_top_k: int = Field(
        default=50, description="Top-k for sub-talker sampling."
    )
    top_p: float = Field(
        default=1, description="Top-p sampling parameter."
    )
    prompt: str = Field(
        default="", description="Optional prompt to guide the style of the generated speech."
    )
    max_new_tokens: int = Field(
        default=200, description="Maximum number of new codec tokens to generate."
    )
    text: str = Field(
        default="", description="The text to be converted to speech."
    )
    language: Language = Field(
        default=Language.AUTO, description="The language of the voice to be designed."
    )
    top_k: int = Field(
        default=50, description="Top-k sampling parameter."
    )
    subtalker_dosample: bool = Field(
        default=True, description="Sampling switch for the sub-talker."
    )
    subtalker_temperature: float = Field(
        default=0.9, description="Temperature for sub-talker sampling."
    )
    subtalker_top_p: float = Field(
        default=1, description="Top-p for sub-talker sampling."
    )
    temperature: float = Field(
        default=0.9, description="Sampling temperature; higher => more random."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "repetition_penalty": self.repetition_penalty,
            "subtalker_top_k": self.subtalker_top_k,
            "top_p": self.top_p,
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "text": self.text,
            "language": self.language.value,
            "top_k": self.top_k,
            "subtalker_dosample": self.subtalker_dosample,
            "subtalker_temperature": self.subtalker_temperature,
            "subtalker_top_p": self.subtalker_top_p,
            "temperature": self.temperature,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-3-tts/voice-design/1.7b",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class Vibevoice05B(FALNode):
    """
    VibeVoice 0.5B generates expressive and emotive speech from text with natural vocal characteristics.
    audio, tts, vibevoice, 0.5b, expressive, text-to-speech

    Use cases:
    - Generate expressive speech from text
    - Create emotive voice-overs
    - Produce natural vocal narration
    - Generate speech with personality
    - Create engaging audio content
    """

    class Speaker(Enum):
        """
        Voice to use for speaking.
        """
        FRANK = "Frank"
        WAYNE = "Wayne"
        CARTER = "Carter"
        EMMA = "Emma"
        GRACE = "Grace"
        MIKE = "Mike"


    script: str = Field(
        default="", description="The script to convert to speech."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation."
    )
    speaker: Speaker = Field(
        default="", description="Voice to use for speaking."
    )
    cfg_scale: float = Field(
        default=1.3, description="CFG (Classifier-Free Guidance) scale for generation. Higher values increase adherence to text."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "script": self.script,
            "seed": self.seed,
            "speaker": self.speaker.value,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vibevoice/0.5b",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class Maya(FALNode):
    """
    Maya generates high-quality natural speech from text with advanced voice synthesis capabilities.
    audio, tts, maya, high-quality, text-to-speech

    Use cases:
    - Generate high-quality speech from text
    - Create professional voice-overs
    - Produce premium audio narration
    - Generate natural-sounding speech
    - Create professional audio content
    """

    class OutputFormat(Enum):
        """
        Output audio format for the generated speech
        """
        WAV = "wav"
        MP3 = "mp3"

    class SampleRate(Enum):
        """
        Output audio sample rate. 48 kHz provides higher quality audio, 24 kHz is faster.
        """
        VALUE_48_KHZ = "48 kHz"
        VALUE_24_KHZ = "24 kHz"


    repetition_penalty: float = Field(
        default=1.1, description="Penalty for repeating tokens. Higher values reduce repetition artifacts."
    )
    prompt: str = Field(
        default="", description="Description of the voice/character. Includes attributes like age, accent, pitch, timbre, pacing, tone, and intensity. See examples for format."
    )
    top_p: float = Field(
        default=0.9, description="Nucleus sampling parameter. Controls diversity of token selection."
    )
    text: str = Field(
        default="", description="The text to synthesize into speech. You can embed emotion tags anywhere in the text using the format <emotion_name>. Available emotions: laugh, laugh_harder, sigh, chuckle, gasp, angry, excited, whisper, cry, scream, sing, snort, exhale, gulp, giggle, sarcastic, curious. Example: 'Hello world! <excited> This is amazing!' or 'I can't believe this <sigh> happened again.'"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.WAV, description="Output audio format for the generated speech"
    )
    max_tokens: int = Field(
        default=2000, description="Maximum number of SNAC tokens to generate (7 tokens per frame). Controls maximum audio length."
    )
    temperature: float = Field(
        default=0.4, description="Sampling temperature. Lower values (0.2-0.5) produce more stable/consistent audio. Higher values add variation."
    )
    sample_rate: SampleRate = Field(
        default=SampleRate.VALUE_48_KHZ, description="Output audio sample rate. 48 kHz provides higher quality audio, 24 kHz is faster."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "repetition_penalty": self.repetition_penalty,
            "prompt": self.prompt,
            "top_p": self.top_p,
            "text": self.text,
            "output_format": self.output_format.value,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "sample_rate": self.sample_rate.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/maya",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MinimaxSpeech26Hd(FALNode):
    """
    Minimax Speech 2.6 HD generates high-definition speech from text with superior audio quality.
    audio, tts, minimax, 2.6, hd, high-quality

    Use cases:
    - Generate HD quality speech from text
    - Create premium voice-overs
    - Produce high-fidelity audio narration
    - Generate superior audio quality speech
    - Create broadcast-quality audio
    """

    class LanguageBoost(Enum):
        """
        Enhance recognition of specified languages and dialects
        """
        CHINESE = "Chinese"
        CHINESE_YUE = "Chinese,Yue"
        ENGLISH = "English"
        ARABIC = "Arabic"
        RUSSIAN = "Russian"
        SPANISH = "Spanish"
        FRENCH = "French"
        PORTUGUESE = "Portuguese"
        GERMAN = "German"
        TURKISH = "Turkish"
        DUTCH = "Dutch"
        UKRAINIAN = "Ukrainian"
        VIETNAMESE = "Vietnamese"
        INDONESIAN = "Indonesian"
        JAPANESE = "Japanese"
        ITALIAN = "Italian"
        KOREAN = "Korean"
        THAI = "Thai"
        POLISH = "Polish"
        ROMANIAN = "Romanian"
        GREEK = "Greek"
        CZECH = "Czech"
        FINNISH = "Finnish"
        HINDI = "Hindi"
        BULGARIAN = "Bulgarian"
        DANISH = "Danish"
        HEBREW = "Hebrew"
        MALAY = "Malay"
        SLOVAK = "Slovak"
        SWEDISH = "Swedish"
        CROATIAN = "Croatian"
        HUNGARIAN = "Hungarian"
        NORWEGIAN = "Norwegian"
        SLOVENIAN = "Slovenian"
        CATALAN = "Catalan"
        NYNORSK = "Nynorsk"
        AFRIKAANS = "Afrikaans"
        AUTO = "auto"

    class MinimaxSpeech26HdOutputFormat(Enum):
        """
        Format of the output content (non-streaming only)
        """
        URL = "url"
        HEX = "hex"


    prompt: str = Field(
        default="", description="Text to convert to speech. Paragraph breaks should be marked with newline characters. **NOTE**: You can customize speech pauses by adding markers in the form `<#x#>`, where `x` is the pause duration in seconds. Valid range: `[0.01, 99.99]`, up to two decimal places. Pause markers must be placed between speakable text segments and cannot be used consecutively."
    )
    language_boost: LanguageBoost | None = Field(
        default=None, description="Enhance recognition of specified languages and dialects"
    )
    output_format: MinimaxSpeech26HdOutputFormat = Field(
        default=MinimaxSpeech26HdOutputFormat.HEX, description="Format of the output content (non-streaming only)"
    )
    pronunciation_dict: str = Field(
        default="", description="Custom pronunciation dictionary for text replacement"
    )
    voice_setting: str = Field(
        default="", description="Voice configuration settings"
    )
    normalization_setting: str = Field(
        default="", description="Loudness normalization settings for the audio"
    )
    audio_setting: str = Field(
        default="", description="Audio configuration settings"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "language_boost": self.language_boost.value if self.language_boost else None,
            "output_format": self.output_format.value,
            "pronunciation_dict": self.pronunciation_dict,
            "voice_setting": self.voice_setting,
            "normalization_setting": self.normalization_setting,
            "audio_setting": self.audio_setting,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/speech-2.6-hd",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MinimaxSpeech26Turbo(FALNode):
    """
    Minimax Speech 2.6 Turbo generates speech from text with optimized speed and good quality.
    audio, tts, minimax, 2.6, turbo, fast

    Use cases:
    - Generate speech quickly from text
    - Create fast voice-overs
    - Produce rapid audio narration
    - Generate speech with turbo speed
    - Create efficient audio content
    """

    class LanguageBoost(Enum):
        """
        Enhance recognition of specified languages and dialects
        """
        CHINESE = "Chinese"
        CHINESE_YUE = "Chinese,Yue"
        ENGLISH = "English"
        ARABIC = "Arabic"
        RUSSIAN = "Russian"
        SPANISH = "Spanish"
        FRENCH = "French"
        PORTUGUESE = "Portuguese"
        GERMAN = "German"
        TURKISH = "Turkish"
        DUTCH = "Dutch"
        UKRAINIAN = "Ukrainian"
        VIETNAMESE = "Vietnamese"
        INDONESIAN = "Indonesian"
        JAPANESE = "Japanese"
        ITALIAN = "Italian"
        KOREAN = "Korean"
        THAI = "Thai"
        POLISH = "Polish"
        ROMANIAN = "Romanian"
        GREEK = "Greek"
        CZECH = "Czech"
        FINNISH = "Finnish"
        HINDI = "Hindi"
        BULGARIAN = "Bulgarian"
        DANISH = "Danish"
        HEBREW = "Hebrew"
        MALAY = "Malay"
        SLOVAK = "Slovak"
        SWEDISH = "Swedish"
        CROATIAN = "Croatian"
        HUNGARIAN = "Hungarian"
        NORWEGIAN = "Norwegian"
        SLOVENIAN = "Slovenian"
        CATALAN = "Catalan"
        NYNORSK = "Nynorsk"
        AFRIKAANS = "Afrikaans"
        AUTO = "auto"

    class MinimaxSpeech26TurboOutputFormat(Enum):
        """
        Format of the output content (non-streaming only)
        """
        URL = "url"
        HEX = "hex"


    prompt: str = Field(
        default="", description="Text to convert to speech. Paragraph breaks should be marked with newline characters. **NOTE**: You can customize speech pauses by adding markers in the form `<#x#>`, where `x` is the pause duration in seconds. Valid range: `[0.01, 99.99]`, up to two decimal places. Pause markers must be placed between speakable text segments and cannot be used consecutively."
    )
    language_boost: LanguageBoost | None = Field(
        default=None, description="Enhance recognition of specified languages and dialects"
    )
    output_format: MinimaxSpeech26TurboOutputFormat = Field(
        default=MinimaxSpeech26TurboOutputFormat.HEX, description="Format of the output content (non-streaming only)"
    )
    pronunciation_dict: str = Field(
        default="", description="Custom pronunciation dictionary for text replacement"
    )
    voice_setting: str = Field(
        default="", description="Voice configuration settings"
    )
    normalization_setting: str = Field(
        default="", description="Loudness normalization settings for the audio"
    )
    audio_setting: str = Field(
        default="", description="Audio configuration settings"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "language_boost": self.language_boost.value if self.language_boost else None,
            "output_format": self.output_format.value,
            "pronunciation_dict": self.pronunciation_dict,
            "voice_setting": self.voice_setting,
            "normalization_setting": self.normalization_setting,
            "audio_setting": self.audio_setting,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/speech-2.6-turbo",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MayaBatch(FALNode):
    """
    Maya Batch TTS generates high-quality speech in batch mode for efficient processing.
    speech, synthesis, text-to-speech, tts, batch, maya

    Use cases:
    - Generate speech for multiple texts
    - Batch process narration
    - Create bulk voice-overs
    - Efficient audio content creation
    - Generate multiple speech files
    """

    class OutputFormat(Enum):
        """
        Output audio format for all generated speech files
        """
        WAV = "wav"
        MP3 = "mp3"

    class SampleRate(Enum):
        """
        Output audio sample rate for all generations. 48 kHz provides higher quality, 24 kHz is faster.
        """
        VALUE_48_KHZ = "48 kHz"
        VALUE_24_KHZ = "24 kHz"


    repetition_penalty: float = Field(
        default=1.1, description="Repetition penalty for all generations."
    )
    top_p: float = Field(
        default=0.9, description="Nucleus sampling parameter for all generations."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.WAV, description="Output audio format for all generated speech files"
    )
    texts: list[str] = Field(
        default=[], description="List of texts to synthesize into speech. You can embed emotion tags in each text using the format <emotion_name>."
    )
    prompts: list[str] = Field(
        default=[], description="List of voice descriptions for each text. Must match the length of texts list. Each describes the voice/character attributes."
    )
    max_tokens: int = Field(
        default=2000, description="Maximum SNAC tokens per generation."
    )
    temperature: float = Field(
        default=0.4, description="Sampling temperature for all generations."
    )
    sample_rate: SampleRate = Field(
        default=SampleRate.VALUE_48_KHZ, description="Output audio sample rate for all generations. 48 kHz provides higher quality, 24 kHz is faster."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "repetition_penalty": self.repetition_penalty,
            "top_p": self.top_p,
            "output_format": self.output_format.value,
            "texts": self.texts,
            "prompts": self.prompts,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "sample_rate": self.sample_rate.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/maya/batch",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MayaStream(FALNode):
    """
    Maya Stream TTS generates high-quality speech in streaming mode for real-time applications.
    speech, synthesis, text-to-speech, tts, streaming, maya

    Use cases:
    - Generate speech in real-time
    - Stream narration dynamically
    - Create live voice-overs
    - Real-time audio synthesis
    - Generate streaming speech
    """

    class OutputFormat(Enum):
        """
        Output audio format. 'mp3' for browser-playable audio, 'wav' for uncompressed audio, 'pcm' for raw PCM (lowest latency, requires client-side decoding).
        """
        MP3 = "mp3"
        WAV = "wav"
        PCM = "pcm"

    class SampleRate(Enum):
        """
        Output audio sample rate. 48 kHz uses upsampling for higher quality audio, 24 kHz is native SNAC output (faster, lower latency).
        """
        VALUE_48_KHZ = "48 kHz"
        VALUE_24_KHZ = "24 kHz"


    repetition_penalty: float = Field(
        default=1.1, description="Penalty for repeating tokens. Higher values reduce repetition artifacts."
    )
    prompt: str = Field(
        default="", description="Description of the voice/character. Includes attributes like age, accent, pitch, timbre, pacing, tone, and intensity. See examples for format."
    )
    top_p: float = Field(
        default=0.9, description="Nucleus sampling parameter. Controls diversity of token selection."
    )
    text: str = Field(
        default="", description="The text to synthesize into speech. You can embed emotion tags anywhere in the text using the format <emotion_name>. Available emotions: laugh, laugh_harder, sigh, chuckle, gasp, angry, excited, whisper, cry, scream, sing, snort, exhale, gulp, giggle, sarcastic, curious. Example: 'Hello world! <excited> This is amazing!' or 'I can't believe this <sigh> happened again.'"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MP3, description="Output audio format. 'mp3' for browser-playable audio, 'wav' for uncompressed audio, 'pcm' for raw PCM (lowest latency, requires client-side decoding)."
    )
    max_tokens: int = Field(
        default=2000, description="Maximum number of SNAC tokens to generate (7 tokens per frame). Controls maximum audio length."
    )
    temperature: float = Field(
        default=0.4, description="Sampling temperature. Lower values (0.2-0.5) produce more stable/consistent audio. Higher values add variation."
    )
    sample_rate: SampleRate = Field(
        default=SampleRate.VALUE_24_KHZ, description="Output audio sample rate. 48 kHz uses upsampling for higher quality audio, 24 kHz is native SNAC output (faster, lower latency)."
    )

    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
            "repetition_penalty": self.repetition_penalty,
            "prompt": self.prompt,
            "top_p": self.top_p,
            "text": self.text,
            "output_format": self.output_format.value,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "sample_rate": self.sample_rate.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/maya/stream",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class IndexTts2TextToSpeech(FALNode):
    """
    Index TTS 2 generates natural-sounding speech from text with advanced neural synthesis.
    speech, synthesis, text-to-speech, tts, neural

    Use cases:
    - Generate natural speech from text
    - Create voice narration
    - Produce audio books
    - Generate voice-overs
    - Create speech content
    """

    prompt: str = Field(
        default="", description="The speech prompt to generate"
    )
    emotional_strengths: str = Field(
        default="", description="The strengths of individual emotions for fine-grained control."
    )
    strength: float = Field(
        default=1, description="The strength of the emotional style transfer. Higher values result in stronger emotional influence."
    )
    emotional_audio_url: AudioRef = Field(
        default=AudioRef(), description="The emotional reference audio file to extract the style from."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="The audio file to generate the speech from."
    )
    emotion_prompt: str = Field(
        default="", description="The emotional prompt to influence the emotional style. Must be used together with should_use_prompt_for_emotion."
    )
    should_use_prompt_for_emotion: bool = Field(
        default=False, description="Whether to use the `prompt` to calculate emotional strengths, if enabled it will overwrite the `emotional_strengths` values. If `emotion_prompt` is provided, it will be used to instead of `prompt` to extract the emotional style."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "emotional_strengths": self.emotional_strengths,
            "strength": self.strength,
            "emotional_audio_url": self.emotional_audio_url,
            "audio_url": self.audio_url,
            "emotion_prompt": self.emotion_prompt,
            "should_use_prompt_for_emotion": self.should_use_prompt_for_emotion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/index-tts-2/text-to-speech",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class KlingVideoV1Tts(FALNode):
    """
    Generate speech from text prompts and different voices using the Kling TTS model, which leverages advanced AI techniques to create high-quality text-to-speech.
    speech, synthesis, text-to-speech, tts

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    class VoiceId(Enum):
        """
        The voice ID to use for speech synthesis
        """
        GENSHIN_VINDI2 = "genshin_vindi2"
        ZHINEN_XUESHENG = "zhinen_xuesheng"
        AOT = "AOT"
        AI_SHATANG = "ai_shatang"
        GENSHIN_KLEE2 = "genshin_klee2"
        GENSHIN_KIRARA = "genshin_kirara"
        AI_KAIYA = "ai_kaiya"
        OVERSEA_MALE1 = "oversea_male1"
        AI_CHENJIAHAO_712 = "ai_chenjiahao_712"
        GIRLFRIEND_4_SPEECH02 = "girlfriend_4_speech02"
        CHAT1_FEMALE_NEW_3 = "chat1_female_new-3"
        CHAT_0407_5_1 = "chat_0407_5-1"
        CARTOON_BOY_07 = "cartoon-boy-07"
        UK_BOY1 = "uk_boy1"
        CARTOON_GIRL_01 = "cartoon-girl-01"
        PEPPAPIG_PLATFORM = "PeppaPig_platform"
        AI_HUANGZHONG_712 = "ai_huangzhong_712"
        AI_HUANGYAOSHI_712 = "ai_huangyaoshi_712"
        AI_LAOGUOWANG_712 = "ai_laoguowang_712"
        CHENGSHU_JIEJIE = "chengshu_jiejie"
        YOU_PINGJING = "you_pingjing"
        CALM_STORY1 = "calm_story1"
        UK_MAN2 = "uk_man2"
        LAOPOPO_SPEECH02 = "laopopo_speech02"
        HEAINAINAI_SPEECH02 = "heainainai_speech02"
        READER_EN_M_V1 = "reader_en_m-v1"
        COMMERCIAL_LADY_EN_F_V1 = "commercial_lady_en_f-v1"
        TIYUXI_XUEDI = "tiyuxi_xuedi"
        TIEXIN_NANYOU = "tiexin_nanyou"
        GIRLFRIEND_1_SPEECH02 = "girlfriend_1_speech02"
        GIRLFRIEND_2_SPEECH02 = "girlfriend_2_speech02"
        ZHUXI_SPEECH02 = "zhuxi_speech02"
        UK_OLDMAN3 = "uk_oldman3"
        DONGBEILAOTIE_SPEECH02 = "dongbeilaotie_speech02"
        CHONGQINGXIAOHUO_SPEECH02 = "chongqingxiaohuo_speech02"
        CHUANMEIZI_SPEECH02 = "chuanmeizi_speech02"
        CHAOSHANDASHU_SPEECH02 = "chaoshandashu_speech02"
        AI_TAIWAN_MAN2_SPEECH02 = "ai_taiwan_man2_speech02"
        XIANZHANGGUI_SPEECH02 = "xianzhanggui_speech02"
        TIANJINJIEJIE_SPEECH02 = "tianjinjiejie_speech02"
        DIYINNANSANG_DB_CN_M_04_V2 = "diyinnansang_DB_CN_M_04-v2"
        YIZHIPIANNAN_V1 = "yizhipiannan-v1"
        GUANXIAOFANG_V2 = "guanxiaofang-v2"
        TIANMEIXUEMEI_V1 = "tianmeixuemei-v1"
        DAOPIANYANSANG_V1 = "daopianyansang-v1"
        MENGWA_V1 = "mengwa-v1"


    text: str = Field(
        default="", description="The text to be converted to speech"
    )
    voice_id: VoiceId = Field(
        default=VoiceId.GENSHIN_VINDI2, description="The voice ID to use for speech synthesis"
    )
    voice_speed: float = Field(
        default=1, description="Rate of speech"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "voice_id": self.voice_id.value,
            "voice_speed": self.voice_speed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1/tts",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class ChatterboxTextToSpeechMultilingual(FALNode):
    """
    Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. Use the first tts from resemble ai.
    speech, synthesis, text-to-speech, tts

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    class CustomAudioLanguage(Enum):
        """
        If using a custom audio URL, specify the language of the audio here. Ignored if voice is not a custom url.
        """
        ENGLISH = "english"
        ARABIC = "arabic"
        DANISH = "danish"
        GERMAN = "german"
        GREEK = "greek"
        SPANISH = "spanish"
        FINNISH = "finnish"
        FRENCH = "french"
        HEBREW = "hebrew"
        HINDI = "hindi"
        ITALIAN = "italian"
        JAPANESE = "japanese"
        KOREAN = "korean"
        MALAY = "malay"
        DUTCH = "dutch"
        NORWEGIAN = "norwegian"
        POLISH = "polish"
        PORTUGUESE = "portuguese"
        RUSSIAN = "russian"
        SWEDISH = "swedish"
        SWAHILI = "swahili"
        TURKISH = "turkish"
        CHINESE = "chinese"


    text: str = Field(
        default="", description="The text to be converted to speech (maximum 300 characters). Supports 23 languages including English, French, German, Spanish, Italian, Portuguese, Hindi, Arabic, Chinese, Japanese, Korean, and more."
    )
    custom_audio_language: CustomAudioLanguage | None = Field(
        default=None, description="If using a custom audio URL, specify the language of the audio here. Ignored if voice is not a custom url."
    )
    exaggeration: float = Field(
        default=0.5, description="Controls speech expressiveness and emotional intensity (0.25-2.0). 0.5 is neutral, higher values increase expressiveness. Extreme values may be unstable."
    )
    voice: str = Field(
        default="english", description="Language code for synthesis. In case using custom please provide audio url and select custom_audio_language."
    )
    temperature: float = Field(
        default=0.8, description="Controls randomness and variation in generation (0.05-5.0). Higher values create more varied speech patterns."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible results. Set to 0 for random generation, or provide a specific number for consistent outputs."
    )
    cfg_scale: float = Field(
        default=0.5, description="Configuration/pace weight controlling generation guidance (0.0-1.0). Use 0.0 for language transfer to mitigate accent inheritance."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "custom_audio_language": self.custom_audio_language.value if self.custom_audio_language else None,
            "exaggeration": self.exaggeration,
            "voice": self.voice,
            "temperature": self.temperature,
            "seed": self.seed,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/chatterbox/text-to-speech/multilingual",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class Vibevoice7b(FALNode):
    """
    Generate long, expressive multi-voice speech using Microsoft's powerful TTS
    speech, synthesis, text-to-speech, tts

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    script: str = Field(
        default="", description="The script to convert to speech. Can be formatted with 'Speaker X:' prefixes for multi-speaker dialogues."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation."
    )
    speakers: list[VibeVoiceSpeaker] = Field(
        default=[], description="List of speakers to use for the script. If not provided, will be inferred from the script or voice samples."
    )
    cfg_scale: float = Field(
        default=1.3, description="CFG (Classifier-Free Guidance) scale for generation. Higher values increase adherence to text."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "script": self.script,
            "seed": self.seed,
            "speakers": self.speakers,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vibevoice/7b",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class Vibevoice(FALNode):
    """
    Generate long, expressive multi-voice speech using Microsoft's powerful TTS
    speech, synthesis, text-to-speech, tts

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    script: str = Field(
        default="", description="The script to convert to speech. Can be formatted with 'Speaker X:' prefixes for multi-speaker dialogues."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation."
    )
    speakers: list[VibeVoiceSpeaker] = Field(
        default=[], description="List of speakers to use for the script. If not provided, will be inferred from the script or voice samples."
    )
    cfg_scale: float = Field(
        default=1.3, description="CFG (Classifier-Free Guidance) scale for generation. Higher values increase adherence to text."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "script": self.script,
            "seed": self.seed,
            "speakers": self.speakers,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vibevoice",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MinimaxPreviewSpeech25Hd(FALNode):
    """
    Generate speech from text prompts and different voices using the MiniMax Speech-02 HD model, which leverages advanced AI techniques to create high-quality text-to-speech.
    speech, synthesis, text-to-speech, tts

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    class LanguageBoost(Enum):
        """
        Enhance recognition of specified languages and dialects
        """
        PERSIAN = "Persian"
        FILIPINO = "Filipino"
        TAMIL = "Tamil"
        CHINESE = "Chinese"
        CHINESE_YUE = "Chinese,Yue"
        ENGLISH = "English"
        ARABIC = "Arabic"
        RUSSIAN = "Russian"
        SPANISH = "Spanish"
        FRENCH = "French"
        PORTUGUESE = "Portuguese"
        GERMAN = "German"
        TURKISH = "Turkish"
        DUTCH = "Dutch"
        UKRAINIAN = "Ukrainian"
        VIETNAMESE = "Vietnamese"
        INDONESIAN = "Indonesian"
        JAPANESE = "Japanese"
        ITALIAN = "Italian"
        KOREAN = "Korean"
        THAI = "Thai"
        POLISH = "Polish"
        ROMANIAN = "Romanian"
        GREEK = "Greek"
        CZECH = "Czech"
        FINNISH = "Finnish"
        HINDI = "Hindi"
        BULGARIAN = "Bulgarian"
        DANISH = "Danish"
        HEBREW = "Hebrew"
        MALAY = "Malay"
        SLOVAK = "Slovak"
        SWEDISH = "Swedish"
        CROATIAN = "Croatian"
        HUNGARIAN = "Hungarian"
        NORWEGIAN = "Norwegian"
        SLOVENIAN = "Slovenian"
        CATALAN = "Catalan"
        NYNORSK = "Nynorsk"
        AFRIKAANS = "Afrikaans"
        AUTO = "auto"

    class OutputFormat(Enum):
        """
        Format of the output content (non-streaming only)
        """
        URL = "url"
        HEX = "hex"


    text: str = Field(
        default="", description="Text to convert to speech (max 5000 characters, minimum 1 non-whitespace character)"
    )
    voice_setting: str = Field(
        default="", description="Voice configuration settings"
    )
    language_boost: LanguageBoost | None = Field(
        default=None, description="Enhance recognition of specified languages and dialects"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.HEX, description="Format of the output content (non-streaming only)"
    )
    pronunciation_dict: str = Field(
        default="", description="Custom pronunciation dictionary for text replacement"
    )
    audio_setting: str = Field(
        default="", description="Audio configuration settings"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "voice_setting": self.voice_setting,
            "language_boost": self.language_boost.value if self.language_boost else None,
            "output_format": self.output_format.value,
            "pronunciation_dict": self.pronunciation_dict,
            "audio_setting": self.audio_setting,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/preview/speech-2.5-hd",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MinimaxPreviewSpeech25Turbo(FALNode):
    """
    Generate fast speech from text prompts and different voices using the MiniMax Speech-02 Turbo model, which leverages advanced AI techniques to create high-quality text-to-speech.
    speech, synthesis, text-to-speech, tts, fast

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    class LanguageBoost(Enum):
        """
        Enhance recognition of specified languages and dialects
        """
        PERSIAN = "Persian"
        FILIPINO = "Filipino"
        TAMIL = "Tamil"
        CHINESE = "Chinese"
        CHINESE_YUE = "Chinese,Yue"
        ENGLISH = "English"
        ARABIC = "Arabic"
        RUSSIAN = "Russian"
        SPANISH = "Spanish"
        FRENCH = "French"
        PORTUGUESE = "Portuguese"
        GERMAN = "German"
        TURKISH = "Turkish"
        DUTCH = "Dutch"
        UKRAINIAN = "Ukrainian"
        VIETNAMESE = "Vietnamese"
        INDONESIAN = "Indonesian"
        JAPANESE = "Japanese"
        ITALIAN = "Italian"
        KOREAN = "Korean"
        THAI = "Thai"
        POLISH = "Polish"
        ROMANIAN = "Romanian"
        GREEK = "Greek"
        CZECH = "Czech"
        FINNISH = "Finnish"
        HINDI = "Hindi"
        BULGARIAN = "Bulgarian"
        DANISH = "Danish"
        HEBREW = "Hebrew"
        MALAY = "Malay"
        SLOVAK = "Slovak"
        SWEDISH = "Swedish"
        CROATIAN = "Croatian"
        HUNGARIAN = "Hungarian"
        NORWEGIAN = "Norwegian"
        SLOVENIAN = "Slovenian"
        CATALAN = "Catalan"
        NYNORSK = "Nynorsk"
        AFRIKAANS = "Afrikaans"
        AUTO = "auto"

    class OutputFormat(Enum):
        """
        Format of the output content (non-streaming only)
        """
        URL = "url"
        HEX = "hex"


    text: str = Field(
        default="", description="Text to convert to speech (max 5000 characters, minimum 1 non-whitespace character)"
    )
    voice_setting: str = Field(
        default="", description="Voice configuration settings"
    )
    language_boost: LanguageBoost | None = Field(
        default=None, description="Enhance recognition of specified languages and dialects"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.HEX, description="Format of the output content (non-streaming only)"
    )
    pronunciation_dict: str = Field(
        default="", description="Custom pronunciation dictionary for text replacement"
    )
    audio_setting: str = Field(
        default="", description="Audio configuration settings"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "voice_setting": self.voice_setting,
            "language_boost": self.language_boost.value if self.language_boost else None,
            "output_format": self.output_format.value,
            "pronunciation_dict": self.pronunciation_dict,
            "audio_setting": self.audio_setting,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/preview/speech-2.5-turbo",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MinimaxVoiceDesign(FALNode):
    """
    Design a personalized voice from a text description, and generate speech from text prompts using the MiniMax model, which leverages advanced AI techniques to create high-quality text-to-speech.
    speech, synthesis, text-to-speech, tts

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    prompt: str = Field(
        default="", description="Voice description prompt for generating a personalized voice"
    )
    preview_text: str = Field(
        default="", description="Text for audio preview. Limited to 500 characters. A fee of $30 per 1M characters will be charged for the generation of the preview audio."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "preview_text": self.preview_text,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/voice-design",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class ResembleAiChatterboxhdTextToSpeech(FALNode):
    """
    Generate expressive, natural speech with Resemble AI's Chatterbox. Features unique emotion control, instant voice cloning from short audio, and built-in watermarking.
    speech, synthesis, text-to-speech, tts

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    class Voice(Enum):
        """
        The voice to use for the TTS request. If neither voice nor audio are provided, a random voice will be used.
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


    text: str = Field(
        default="My name is Maximus Decimus Meridius, commander of the Armies of the North, General of the Felix Legions and loyal servant to the true emperor, Marcus Aurelius. Father to a murdered son, husband to a murdered wife. And I will have my vengeance, in this life or the next.", description="Text to synthesize into speech."
    )
    exaggeration: float = Field(
        default=0.5, description="Controls emotion exaggeration. Range typically 0.25 to 2.0."
    )
    high_quality_audio: bool = Field(
        default=False, description="If True, the generated audio will be upscaled to 48kHz. The generation of the audio will take longer, but the quality will be higher. If False, the generated audio will be 24kHz."
    )
    voice: Voice | None = Field(
        default=None, description="The voice to use for the TTS request. If neither voice nor audio are provided, a random voice will be used."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL to the audio sample to use as a voice prompt for zero-shot TTS voice cloning. Providing a audio sample will override the voice setting. If neither voice nor audio_url are provided, a random voice will be used."
    )
    temperature: float = Field(
        default=0.8, description="Controls the randomness of generation. Range typically 0.05 to 5."
    )
    seed: int = Field(
        default=0, description="Useful to control the reproducibility of the generated audio. Assuming all other properties didn't change, a fixed seed should always generate the exact same audio file. Set to 0 for random seed."
    )
    cfg: float = Field(
        default=0.5, description="Classifier-free guidance scale (CFG) controls the conditioning factor. Range typically 0.2 to 1.0. For expressive or dramatic speech, try lower cfg values (e.g. ~0.3) and increase exaggeration to around 0.7 or higher. If the reference speaker has a fast speaking style, lowering cfg to around 0.3 can improve pacing."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "exaggeration": self.exaggeration,
            "high_quality_audio": self.high_quality_audio,
            "voice": self.voice.value if self.voice else None,
            "audio_url": self.audio_url,
            "temperature": self.temperature,
            "seed": self.seed,
            "cfg": self.cfg,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="resemble-ai/chatterboxhd/text-to-speech",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class ChatterboxTextToSpeech(FALNode):
    """
    Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. Use the first tts from resemble ai.
    speech, synthesis, text-to-speech, tts

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    text: str = Field(
        default="", description="The text to be converted to speech. You can additionally add the following emotive tags: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>"
    )
    exaggeration: float = Field(
        default=0.25, description="Exaggeration factor for the generated speech (0.0 = no exaggeration, 1.0 = maximum exaggeration)."
    )
    audio_url: AudioRef = Field(
        default="https://storage.googleapis.com/chatterbox-demo-samples/prompts/male_rickmorty.mp3", description="Optional URL to an audio file to use as a reference for the generated speech. If provided, the model will try to match the style and tone of the reference audio."
    )
    temperature: float = Field(
        default=0.7, description="Temperature for generation (higher = more creative)."
    )
    seed: int = Field(
        default=-1, description="Useful to control the reproducibility of the generated audio. Assuming all other properties didn't change, a fixed seed should always generate the exact same audio file. Set to 0 for random seed.."
    )
    cfg: float = Field(
        default=0.5
    )

    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
            "text": self.text,
            "exaggeration": self.exaggeration,
            "audio_url": self.audio_url,
            "temperature": self.temperature,
            "seed": self.seed,
            "cfg": self.cfg,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/chatterbox/text-to-speech",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MinimaxVoiceClone(FALNode):
    """
    Clone a voice from a sample audio and generate speech from text prompts using the MiniMax model, which leverages advanced AI techniques to create high-quality text-to-speech.
    speech, synthesis, text-to-speech, tts

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    class Model(Enum):
        """
        TTS model to use for preview. Options: speech-02-hd, speech-02-turbo, speech-01-hd, speech-01-turbo
        """
        SPEECH_02_HD = "speech-02-hd"
        SPEECH_02_TURBO = "speech-02-turbo"
        SPEECH_01_HD = "speech-01-hd"
        SPEECH_01_TURBO = "speech-01-turbo"


    model: Model = Field(
        default=Model.SPEECH_02_HD, description="TTS model to use for preview. Options: speech-02-hd, speech-02-turbo, speech-01-hd, speech-01-turbo"
    )
    text: str = Field(
        default="Hello, this is a preview of your cloned voice! I hope you like it!", description="Text to generate a TTS preview with the cloned voice (optional)"
    )
    accuracy: float = Field(
        default=0.0, description="Text validation accuracy threshold (0-1)"
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL of the input audio file for voice cloning. Should be at least 10 seconds long. To retain the voice permanently, use it with a TTS (text-to-speech) endpoint at least once within 7 days. Otherwise, it will be automatically deleted."
    )
    noise_reduction: bool = Field(
        default=False, description="Enable noise reduction for the cloned voice"
    )
    need_volume_normalization: bool = Field(
        default=False, description="Enable volume normalization for the cloned voice"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "model": self.model.value,
            "text": self.text,
            "accuracy": self.accuracy,
            "audio_url": self.audio_url,
            "noise_reduction": self.noise_reduction,
            "need_volume_normalization": self.need_volume_normalization,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/voice-clone",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MinimaxSpeech02Turbo(FALNode):
    """
    Generate fast speech from text prompts and different voices using the MiniMax Speech-02 Turbo model, which leverages advanced AI techniques to create high-quality text-to-speech.
    speech, synthesis, text-to-speech, tts, fast

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    class LanguageBoost(Enum):
        """
        Enhance recognition of specified languages and dialects
        """
        CHINESE = "Chinese"
        CHINESE_YUE = "Chinese,Yue"
        ENGLISH = "English"
        ARABIC = "Arabic"
        RUSSIAN = "Russian"
        SPANISH = "Spanish"
        FRENCH = "French"
        PORTUGUESE = "Portuguese"
        GERMAN = "German"
        TURKISH = "Turkish"
        DUTCH = "Dutch"
        UKRAINIAN = "Ukrainian"
        VIETNAMESE = "Vietnamese"
        INDONESIAN = "Indonesian"
        JAPANESE = "Japanese"
        ITALIAN = "Italian"
        KOREAN = "Korean"
        THAI = "Thai"
        POLISH = "Polish"
        ROMANIAN = "Romanian"
        GREEK = "Greek"
        CZECH = "Czech"
        FINNISH = "Finnish"
        HINDI = "Hindi"
        BULGARIAN = "Bulgarian"
        DANISH = "Danish"
        HEBREW = "Hebrew"
        MALAY = "Malay"
        SLOVAK = "Slovak"
        SWEDISH = "Swedish"
        CROATIAN = "Croatian"
        HUNGARIAN = "Hungarian"
        NORWEGIAN = "Norwegian"
        SLOVENIAN = "Slovenian"
        CATALAN = "Catalan"
        NYNORSK = "Nynorsk"
        AFRIKAANS = "Afrikaans"
        AUTO = "auto"

    class OutputFormat(Enum):
        """
        Format of the output content (non-streaming only)
        """
        URL = "url"
        HEX = "hex"


    text: str = Field(
        default="", description="Text to convert to speech (max 5000 characters, minimum 1 non-whitespace character)"
    )
    voice_setting: str = Field(
        default="", description="Voice configuration settings"
    )
    language_boost: LanguageBoost | None = Field(
        default=None, description="Enhance recognition of specified languages and dialects"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.HEX, description="Format of the output content (non-streaming only)"
    )
    pronunciation_dict: str = Field(
        default="", description="Custom pronunciation dictionary for text replacement"
    )
    audio_setting: str = Field(
        default="", description="Audio configuration settings"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "voice_setting": self.voice_setting,
            "language_boost": self.language_boost.value if self.language_boost else None,
            "output_format": self.output_format.value,
            "pronunciation_dict": self.pronunciation_dict,
            "audio_setting": self.audio_setting,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/speech-02-turbo",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MinimaxSpeech02Hd(FALNode):
    """
    Generate speech from text prompts and different voices using the MiniMax Speech-02 HD model, which leverages advanced AI techniques to create high-quality text-to-speech.
    speech, synthesis, text-to-speech, tts

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    class LanguageBoost(Enum):
        """
        Enhance recognition of specified languages and dialects
        """
        CHINESE = "Chinese"
        CHINESE_YUE = "Chinese,Yue"
        ENGLISH = "English"
        ARABIC = "Arabic"
        RUSSIAN = "Russian"
        SPANISH = "Spanish"
        FRENCH = "French"
        PORTUGUESE = "Portuguese"
        GERMAN = "German"
        TURKISH = "Turkish"
        DUTCH = "Dutch"
        UKRAINIAN = "Ukrainian"
        VIETNAMESE = "Vietnamese"
        INDONESIAN = "Indonesian"
        JAPANESE = "Japanese"
        ITALIAN = "Italian"
        KOREAN = "Korean"
        THAI = "Thai"
        POLISH = "Polish"
        ROMANIAN = "Romanian"
        GREEK = "Greek"
        CZECH = "Czech"
        FINNISH = "Finnish"
        HINDI = "Hindi"
        BULGARIAN = "Bulgarian"
        DANISH = "Danish"
        HEBREW = "Hebrew"
        MALAY = "Malay"
        SLOVAK = "Slovak"
        SWEDISH = "Swedish"
        CROATIAN = "Croatian"
        HUNGARIAN = "Hungarian"
        NORWEGIAN = "Norwegian"
        SLOVENIAN = "Slovenian"
        CATALAN = "Catalan"
        NYNORSK = "Nynorsk"
        AFRIKAANS = "Afrikaans"
        AUTO = "auto"

    class OutputFormat(Enum):
        """
        Format of the output content (non-streaming only)
        """
        URL = "url"
        HEX = "hex"


    text: str = Field(
        default="", description="Text to convert to speech (max 5000 characters, minimum 1 non-whitespace character)"
    )
    voice_setting: str = Field(
        default="", description="Voice configuration settings"
    )
    language_boost: LanguageBoost | None = Field(
        default=None, description="Enhance recognition of specified languages and dialects"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.HEX, description="Format of the output content (non-streaming only)"
    )
    pronunciation_dict: str = Field(
        default="", description="Custom pronunciation dictionary for text replacement"
    )
    audio_setting: str = Field(
        default="", description="Audio configuration settings"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "voice_setting": self.voice_setting,
            "language_boost": self.language_boost.value if self.language_boost else None,
            "output_format": self.output_format.value,
            "pronunciation_dict": self.pronunciation_dict,
            "audio_setting": self.audio_setting,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/speech-02-hd",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class DiaTts(FALNode):
    """
    Dia directly generates realistic dialogue from transcripts. Audio conditioning enables emotion control. Produces natural nonverbals like laughter and throat clearing.
    speech, synthesis, text-to-speech, tts

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    text: str = Field(
        default="", description="The text to be converted to speech."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/dia-tts",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class OrpheusTts(FALNode):
    """
    Orpheus TTS is a state-of-the-art, Llama-based Speech-LLM designed for high-quality, empathetic text-to-speech generation. This model has been finetuned to deliver human-level speech synthesis, achieving exceptional clarity, expressiveness, and real-time performances.
    speech, synthesis, text-to-speech, tts

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    class Voice(Enum):
        """
        Voice ID for the desired voice.
        """
        TARA = "tara"
        LEAH = "leah"
        JESS = "jess"
        LEO = "leo"
        DAN = "dan"
        MIA = "mia"
        ZAC = "zac"
        ZOE = "zoe"


    text: str = Field(
        default="", description="The text to be converted to speech. You can additionally add the following emotive tags: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>"
    )
    voice: Voice = Field(
        default=Voice.TARA, description="Voice ID for the desired voice."
    )
    repetition_penalty: float = Field(
        default=1.2, description="Repetition penalty (>= 1.1 required for stable generations)."
    )
    temperature: float = Field(
        default=0.7, description="Temperature for generation (higher = more creative)."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "voice": self.voice.value,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/orpheus-tts",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class ElevenlabsTtsTurboV25(FALNode):
    """
    Generate high-speed text-to-speech audio using ElevenLabs TTS Turbo v2.5.
    speech, synthesis, text-to-speech, tts, fast

    Use cases:
    - Voice synthesis for applications
    - Audiobook narration
    - Virtual assistant voices
    - Accessibility solutions
    - Content localization
    """

    class ApplyTextNormalization(Enum):
        """
        This parameter controls text normalization with three modes: 'auto', 'on', and 'off'. When set to 'auto', the system will automatically decide whether to apply text normalization (e.g., spelling out numbers). With 'on', text normalization will always be applied, while with 'off', it will be skipped.
        """
        AUTO = "auto"
        ON = "on"
        OFF = "off"


    stability: float = Field(
        default=0.5, description="Voice stability (0-1)"
    )
    next_text: str = Field(
        default="", description="The text that comes after the text of the current request. Can be used to improve the speech's continuity when concatenating together multiple generations or to influence the speech's continuity in the current generation."
    )
    speed: float = Field(
        default=1, description="Speech speed (0.7-1.2). Values below 1.0 slow down the speech, above 1.0 speed it up. Extreme values may affect quality."
    )
    style: float = Field(
        default=0, description="Style exaggeration (0-1)"
    )
    text: str = Field(
        default="", description="The text to convert to speech"
    )
    timestamps: bool = Field(
        default=False, description="Whether to return timestamps for each word in the generated speech"
    )
    similarity_boost: float = Field(
        default=0.75, description="Similarity boost (0-1)"
    )
    voice: str = Field(
        default="Rachel", description="The voice to use for speech generation"
    )
    language_code: str = Field(
        default="", description="Language code (ISO 639-1) used to enforce a language for the model. An error will be returned if language code is not supported by the model."
    )
    apply_text_normalization: ApplyTextNormalization = Field(
        default=ApplyTextNormalization.AUTO, description="This parameter controls text normalization with three modes: 'auto', 'on', and 'off'. When set to 'auto', the system will automatically decide whether to apply text normalization (e.g., spelling out numbers). With 'on', text normalization will always be applied, while with 'off', it will be skipped."
    )
    previous_text: str = Field(
        default="", description="The text that came before the text of the current request. Can be used to improve the speech's continuity when concatenating together multiple generations or to influence the speech's continuity in the current generation."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "stability": self.stability,
            "next_text": self.next_text,
            "speed": self.speed,
            "style": self.style,
            "text": self.text,
            "timestamps": self.timestamps,
            "similarity_boost": self.similarity_boost,
            "voice": self.voice,
            "language_code": self.language_code,
            "apply_text_normalization": self.apply_text_normalization.value,
            "previous_text": self.previous_text,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/tts/turbo-v2.5",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]