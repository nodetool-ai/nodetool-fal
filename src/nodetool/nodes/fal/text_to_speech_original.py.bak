from enum import Enum
from pydantic import Field
from nodetool.metadata.types import AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


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


class MinimaxSpeech26TurboOutputFormat(Enum):
    """
    Format of the output content (non-streaming only)
    """
    URL = "url"
    HEX = "hex"




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