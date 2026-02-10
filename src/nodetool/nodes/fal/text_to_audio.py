from enum import Enum
from pydantic import Field
from nodetool.metadata.types import AudioRef
from nodetool.nodes.fal.types import DialogueBlock, InpaintSection, PronunciationDictionaryLocator, Speaker, Turn
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class ACEStepPromptToAudio(FALNode):
    """
    ACE-Step generates music from text prompts with high-quality audio synthesis.
    audio, generation, music, ace-step, text-to-audio

    Use cases:
    - Generate music from text descriptions
    - Create background music for videos
    - Produce royalty-free music
    - Generate audio soundtracks
    - Create custom music compositions
    """

    class Scheduler(Enum):
        """
        Scheduler to use for the generation process.
        """
        EULER = "euler"
        HEUN = "heun"

    class GuidanceType(Enum):
        """
        Type of CFG to use for the generation process.
        """
        CFG = "cfg"
        APG = "apg"
        CFG_STAR = "cfg_star"


    number_of_steps: int = Field(
        default=27, description="Number of steps to generate the audio."
    )
    duration: float = Field(
        default=60, description="The duration of the generated audio in seconds."
    )
    prompt: str = Field(
        default="", description="Prompt to control the style of the generated audio. This will be used to generate tags and lyrics."
    )
    minimum_guidance_scale: float = Field(
        default=3, description="Minimum guidance scale for the generation after the decay."
    )
    tag_guidance_scale: float = Field(
        default=5, description="Tag guidance scale for the generation."
    )
    scheduler: Scheduler = Field(
        default=Scheduler.EULER, description="Scheduler to use for the generation process."
    )
    guidance_scale: float = Field(
        default=15, description="Guidance scale for the generation."
    )
    guidance_type: GuidanceType = Field(
        default=GuidanceType.APG, description="Type of CFG to use for the generation process."
    )
    instrumental: bool = Field(
        default=False, description="Whether to generate an instrumental version of the audio."
    )
    lyric_guidance_scale: float = Field(
        default=1.5, description="Lyric guidance scale for the generation."
    )
    guidance_interval: float = Field(
        default=0.5, description="Guidance interval for the generation. 0.5 means only apply guidance in the middle steps (0.25 * infer_steps to 0.75 * infer_steps)"
    )
    guidance_interval_decay: float = Field(
        default=0, description="Guidance interval decay for the generation. Guidance scale will decay from guidance_scale to min_guidance_scale in the interval. 0.0 means no decay."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If not provided, a random seed will be used."
    )
    granularity_scale: int = Field(
        default=10, description="Granularity scale for the generation process. Higher values can reduce artifacts."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "number_of_steps": self.number_of_steps,
            "duration": self.duration,
            "prompt": self.prompt,
            "minimum_guidance_scale": self.minimum_guidance_scale,
            "tag_guidance_scale": self.tag_guidance_scale,
            "scheduler": self.scheduler.value,
            "guidance_scale": self.guidance_scale,
            "guidance_type": self.guidance_type.value,
            "instrumental": self.instrumental,
            "lyric_guidance_scale": self.lyric_guidance_scale,
            "guidance_interval": self.guidance_interval,
            "guidance_interval_decay": self.guidance_interval_decay,
            "seed": self.seed,
            "granularity_scale": self.granularity_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ace-step/prompt-to-audio",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class ACEStep(FALNode):
    """
    ACE-Step generates music with lyrics from text using advanced audio synthesis.
    audio, generation, music, lyrics, ace-step, text-to-audio

    Use cases:
    - Generate songs with lyrics
    - Create music with vocal tracks
    - Produce complete songs from text
    - Generate lyrical content
    - Create vocal music compositions
    """

    class Scheduler(Enum):
        """
        Scheduler to use for the generation process.
        """
        EULER = "euler"
        HEUN = "heun"

    class GuidanceType(Enum):
        """
        Type of CFG to use for the generation process.
        """
        CFG = "cfg"
        APG = "apg"
        CFG_STAR = "cfg_star"


    number_of_steps: int = Field(
        default=27, description="Number of steps to generate the audio."
    )
    duration: float = Field(
        default=60, description="The duration of the generated audio in seconds."
    )
    tags: str = Field(
        default="", description="Comma-separated list of genre tags to control the style of the generated audio."
    )
    minimum_guidance_scale: float = Field(
        default=3, description="Minimum guidance scale for the generation after the decay."
    )
    lyrics: str = Field(
        default="", description="Lyrics to be sung in the audio. If not provided or if [inst] or [instrumental] is the content of this field, no lyrics will be sung. Use control structures like [verse], [chorus] and [bridge] to control the structure of the song."
    )
    tag_guidance_scale: float = Field(
        default=5, description="Tag guidance scale for the generation."
    )
    scheduler: Scheduler = Field(
        default=Scheduler.EULER, description="Scheduler to use for the generation process."
    )
    guidance_scale: float = Field(
        default=15, description="Guidance scale for the generation."
    )
    guidance_type: GuidanceType = Field(
        default=GuidanceType.APG, description="Type of CFG to use for the generation process."
    )
    lyric_guidance_scale: float = Field(
        default=1.5, description="Lyric guidance scale for the generation."
    )
    guidance_interval: float = Field(
        default=0.5, description="Guidance interval for the generation. 0.5 means only apply guidance in the middle steps (0.25 * infer_steps to 0.75 * infer_steps)"
    )
    guidance_interval_decay: float = Field(
        default=0, description="Guidance interval decay for the generation. Guidance scale will decay from guidance_scale to min_guidance_scale in the interval. 0.0 means no decay."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If not provided, a random seed will be used."
    )
    granularity_scale: int = Field(
        default=10, description="Granularity scale for the generation process. Higher values can reduce artifacts."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "number_of_steps": self.number_of_steps,
            "duration": self.duration,
            "tags": self.tags,
            "minimum_guidance_scale": self.minimum_guidance_scale,
            "lyrics": self.lyrics,
            "tag_guidance_scale": self.tag_guidance_scale,
            "scheduler": self.scheduler.value,
            "guidance_scale": self.guidance_scale,
            "guidance_type": self.guidance_type.value,
            "lyric_guidance_scale": self.lyric_guidance_scale,
            "guidance_interval": self.guidance_interval,
            "guidance_interval_decay": self.guidance_interval_decay,
            "seed": self.seed,
            "granularity_scale": self.granularity_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ace-step",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class CSM1B(FALNode):
    """
    CSM (Conversational Speech Model) generates natural conversational speech from text.
    audio, speech, tts, conversational, text-to-speech

    Use cases:
    - Generate natural conversation audio
    - Create dialogue for characters
    - Produce conversational voice content
    - Generate realistic speech
    - Create interactive voice responses
    """

    scene: list[Turn] = Field(
        default=[], description="The text to generate an audio from."
    )
    context: list[Speaker] = Field(
        default=[], description="The context to generate an audio from."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "scene": [item.model_dump(exclude={"type"}) for item in self.scene],
            "context": [item.model_dump(exclude={"type"}) for item in self.context],
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/csm-1b",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class DiffRhythm(FALNode):
    """
    DiffRhythm generates rhythmic music and beats using diffusion models.
    audio, generation, rhythm, beats, music, text-to-audio

    Use cases:
    - Generate rhythmic music
    - Create drum beats
    - Produce percussion tracks
    - Generate rhythm patterns
    - Create beat sequences
    """

    class MusicDuration(Enum):
        """
        The duration of the music to generate.
        """
        VALUE_95S = "95s"
        VALUE_285S = "285s"

    class Scheduler(Enum):
        """
        The scheduler to use for the music generation.
        """
        EULER = "euler"
        MIDPOINT = "midpoint"
        RK4 = "rk4"
        IMPLICIT_ADAMS = "implicit_adams"


    lyrics: str = Field(
        default="", description="The prompt to generate the song from. Must have two sections. Sections start with either [chorus] or a [verse]."
    )
    cfg_strength: float = Field(
        default=4, description="The CFG strength to use for the music generation."
    )
    reference_audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the reference audio to use for the music generation."
    )
    music_duration: MusicDuration = Field(
        default=MusicDuration.VALUE_95S, description="The duration of the music to generate."
    )
    scheduler: Scheduler = Field(
        default=Scheduler.EULER, description="The scheduler to use for the music generation."
    )
    num_inference_steps: int = Field(
        default=32, description="The number of inference steps to use for the music generation."
    )
    style_prompt: str = Field(
        default="", description="The style prompt to use for the music generation."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "lyrics": self.lyrics,
            "cfg_strength": self.cfg_strength,
            "reference_audio_url": self.reference_audio,
            "music_duration": self.music_duration.value,
            "scheduler": self.scheduler.value,
            "num_inference_steps": self.num_inference_steps,
            "style_prompt": self.style_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/diffrhythm",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class ElevenLabsTTSMultilingualV2(FALNode):
    """
    ElevenLabs Multilingual TTS v2 generates natural speech in multiple languages.
    audio, tts, speech, multilingual, elevenlabs, text-to-speech

    Use cases:
    - Generate multilingual speech
    - Create voiceovers in multiple languages
    - Produce localized audio content
    - Generate international voice content
    - Create translated audio
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
            application="fal-ai/elevenlabs/tts/multilingual-v2",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text", "language"]

class ElevenLabsTextToDialogueV3(FALNode):
    """
    ElevenLabs Text to Dialogue v3 generates conversational dialogue with multiple speakers.
    audio, dialogue, conversation, elevenlabs, text-to-speech

    Use cases:
    - Generate multi-speaker dialogue
    - Create conversational audio
    - Produce podcast-style content
    - Generate character conversations
    - Create interactive dialogues
    """

    stability: str = Field(
        default="", description="Determines how stable the voice is and the randomness between each generation. Lower values introduce broader emotional range for the voice. Higher values can result in a monotonous voice with limited emotion. Must be one of 0.0, 0.5, 1.0, else it will be rounded to the nearest value."
    )
    language_code: str = Field(
        default="", description="Language code (ISO 639-1) used to enforce a language for the model. An error will be returned if language code is not supported by the model."
    )
    inputs: list[DialogueBlock] = Field(
        default=[], description="A list of dialogue inputs, each containing text and a voice ID which will be converted into speech."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility."
    )
    use_speaker_boost: str = Field(
        default="", description="This setting boosts the similarity to the original speaker. Using this setting requires a slightly higher computational load, which in turn increases latency."
    )
    pronunciation_dictionary_locators: list[PronunciationDictionaryLocator] = Field(
        default=[], description="A list of pronunciation dictionary locators (id, version_id) to be applied to the text. They will be applied in order. You may have up to 3 locators per request"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "stability": self.stability,
            "language_code": self.language_code,
            "inputs": [item.model_dump(exclude={"type"}) for item in self.inputs],
            "seed": self.seed,
            "use_speaker_boost": self.use_speaker_boost,
            "pronunciation_dictionary_locators": [item.model_dump(exclude={"type"}) for item in self.pronunciation_dictionary_locators],
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/text-to-dialogue/eleven-v3",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class ElevenLabsSoundEffectsV2(FALNode):
    """
    ElevenLabs Sound Effects v2 generates custom sound effects from text descriptions.
    audio, sound-effects, sfx, elevenlabs, text-to-audio

    Use cases:
    - Generate custom sound effects
    - Create audio effects for videos
    - Produce game sound effects
    - Generate environmental sounds
    - Create audio atmosphere
    """

    class OutputFormat(Enum):
        """
        Output format of the generated audio. Formatted as codec_sample_rate_bitrate.
        """
        MP3_22050_32 = "mp3_22050_32"
        MP3_44100_32 = "mp3_44100_32"
        MP3_44100_64 = "mp3_44100_64"
        MP3_44100_96 = "mp3_44100_96"
        MP3_44100_128 = "mp3_44100_128"
        MP3_44100_192 = "mp3_44100_192"
        PCM_8000 = "pcm_8000"
        PCM_16000 = "pcm_16000"
        PCM_22050 = "pcm_22050"
        PCM_24000 = "pcm_24000"
        PCM_44100 = "pcm_44100"
        PCM_48000 = "pcm_48000"
        ULAW_8000 = "ulaw_8000"
        ALAW_8000 = "alaw_8000"
        OPUS_48000_32 = "opus_48000_32"
        OPUS_48000_64 = "opus_48000_64"
        OPUS_48000_96 = "opus_48000_96"
        OPUS_48000_128 = "opus_48000_128"
        OPUS_48000_192 = "opus_48000_192"


    text: str = Field(
        default="", description="The text describing the sound effect to generate"
    )
    loop: bool = Field(
        default=False, description="Whether to create a sound effect that loops smoothly."
    )
    prompt_influence: float = Field(
        default=0.3, description="How closely to follow the prompt (0-1). Higher values mean less variation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MP3_44100_128, description="Output format of the generated audio. Formatted as codec_sample_rate_bitrate."
    )
    duration_seconds: str = Field(
        default="", description="Duration in seconds (0.5-22). If None, optimal duration will be determined from prompt."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "loop": self.loop,
            "prompt_influence": self.prompt_influence,
            "output_format": self.output_format.value,
            "duration_seconds": self.duration_seconds,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/sound-effects/v2",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class ElevenLabsTTSV3(FALNode):
    """
    ElevenLabs TTS v3 generates high-quality natural speech with advanced voice control.
    audio, tts, speech, elevenlabs, text-to-speech

    Use cases:
    - Generate high-quality voiceovers
    - Create natural speech audio
    - Produce professional narration
    - Generate expressive speech
    - Create audiobook content
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
    speed: float = Field(
        default=1, description="Speech speed (0.7-1.2). Values below 1.0 slow down the speech, above 1.0 speed it up. Extreme values may affect quality."
    )
    text: str = Field(
        default="", description="The text to convert to speech"
    )
    style: float = Field(
        default=0, description="Style exaggeration (0-1)"
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
        default="", description="Language code (ISO 639-1) used to enforce a language for the model."
    )
    apply_text_normalization: ApplyTextNormalization = Field(
        default=ApplyTextNormalization.AUTO, description="This parameter controls text normalization with three modes: 'auto', 'on', and 'off'. When set to 'auto', the system will automatically decide whether to apply text normalization (e.g., spelling out numbers). With 'on', text normalization will always be applied, while with 'off', it will be skipped."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "stability": self.stability,
            "speed": self.speed,
            "text": self.text,
            "style": self.style,
            "timestamps": self.timestamps,
            "similarity_boost": self.similarity_boost,
            "voice": self.voice,
            "language_code": self.language_code,
            "apply_text_normalization": self.apply_text_normalization.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/tts/eleven-v3",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class ElevenLabsMusic(FALNode):
    """
    ElevenLabs Music generates custom music compositions from text descriptions.
    audio, music, generation, elevenlabs, text-to-audio

    Use cases:
    - Generate custom music
    - Create background scores
    - Produce original compositions
    - Generate mood music
    - Create cinematic soundtracks
    """

    class OutputFormat(Enum):
        """
        Output format of the generated audio. Formatted as codec_sample_rate_bitrate. So an mp3 with 22.05kHz sample rate at 32kbs is represented as mp3_22050_32. MP3 with 192kbps bitrate requires you to be subscribed to Creator tier or above. PCM with 44.1kHz sample rate requires you to be subscribed to Pro tier or above. Note that the μ-law format (sometimes written mu-law, often approximated as u-law) is commonly used for Twilio audio inputs.
        """
        MP3_22050_32 = "mp3_22050_32"
        MP3_44100_32 = "mp3_44100_32"
        MP3_44100_64 = "mp3_44100_64"
        MP3_44100_96 = "mp3_44100_96"
        MP3_44100_128 = "mp3_44100_128"
        MP3_44100_192 = "mp3_44100_192"
        PCM_8000 = "pcm_8000"
        PCM_16000 = "pcm_16000"
        PCM_22050 = "pcm_22050"
        PCM_24000 = "pcm_24000"
        PCM_44100 = "pcm_44100"
        PCM_48000 = "pcm_48000"
        ULAW_8000 = "ulaw_8000"
        ALAW_8000 = "alaw_8000"
        OPUS_48000_32 = "opus_48000_32"
        OPUS_48000_64 = "opus_48000_64"
        OPUS_48000_96 = "opus_48000_96"
        OPUS_48000_128 = "opus_48000_128"
        OPUS_48000_192 = "opus_48000_192"


    prompt: str = Field(
        default="", description="The text prompt describing the music to generate"
    )
    composition_plan: str = Field(
        default="", description="The composition plan for the music"
    )
    music_length_ms: str = Field(
        default="", description="The length of the song to generate in milliseconds. Used only in conjunction with prompt. Must be between 3000ms and 600000ms. Optional - if not provided, the model will choose a length based on the prompt."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MP3_44100_128, description="Output format of the generated audio. Formatted as codec_sample_rate_bitrate. So an mp3 with 22.05kHz sample rate at 32kbs is represented as mp3_22050_32. MP3 with 192kbps bitrate requires you to be subscribed to Creator tier or above. PCM with 44.1kHz sample rate requires you to be subscribed to Pro tier or above. Note that the μ-law format (sometimes written mu-law, often approximated as u-law) is commonly used for Twilio audio inputs."
    )
    respect_sections_durations: bool = Field(
        default=True, description="Controls how strictly section durations in the composition_plan are enforced. It will only have an effect if it is used with composition_plan. When set to true, the model will precisely respect each section's duration_ms from the plan. When set to false, the model may adjust individual section durations which will generally lead to better generation quality and improved latency, while always preserving the total song duration from the plan."
    )
    force_instrumental: bool = Field(
        default=False, description="If true, guarantees that the generated song will be instrumental. If false, the song may or may not be instrumental depending on the prompt. Can only be used with prompt."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "composition_plan": self.composition_plan,
            "music_length_ms": self.music_length_ms,
            "output_format": self.output_format.value,
            "respect_sections_durations": self.respect_sections_durations,
            "force_instrumental": self.force_instrumental,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/music",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class F5TTS(FALNode):
    """
    F5 TTS generates natural speech with fast inference and high quality.
    audio, tts, speech, fast, text-to-speech

    Use cases:
    - Fast speech generation
    - Real-time TTS applications
    - Quick voiceover creation
    - Efficient speech synthesis
    - Rapid audio production
    """

    class ModelType(Enum):
        """
        The name of the model to be used for TTS.
        """
        F5_TTS = "F5-TTS"
        E2_TTS = "E2-TTS"


    ref_text: str = Field(
        default="", description="The reference text to be used for TTS. If not provided, an ASR (Automatic Speech Recognition) model will be used to generate the reference text."
    )
    remove_silence: bool = Field(
        default=True, description="Whether to remove the silence from the audio file."
    )
    gen_text: str = Field(
        default="", description="The text to be converted to speech."
    )
    model_type: ModelType = Field(
        default="", description="The name of the model to be used for TTS."
    )
    ref_audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the reference audio file."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "ref_text": self.ref_text,
            "remove_silence": self.remove_silence,
            "gen_text": self.gen_text,
            "model_type": self.model_type.value,
            "ref_audio_url": self.ref_audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/f5-tts",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class Kokoro(FALNode):
    """
    Kokoro generates expressive and emotional speech with advanced prosody control.
    audio, tts, speech, expressive, emotional, text-to-speech

    Use cases:
    - Generate expressive speech
    - Create emotional voiceovers
    - Produce dramatic narration
    - Generate character voices
    - Create emotive audio content
    """

    class Voice(Enum):
        """
        Voice ID for the desired voice.
        """
        AF_HEART = "af_heart"
        AF_ALLOY = "af_alloy"
        AF_AOEDE = "af_aoede"
        AF_BELLA = "af_bella"
        AF_JESSICA = "af_jessica"
        AF_KORE = "af_kore"
        AF_NICOLE = "af_nicole"
        AF_NOVA = "af_nova"
        AF_RIVER = "af_river"
        AF_SARAH = "af_sarah"
        AF_SKY = "af_sky"
        AM_ADAM = "am_adam"
        AM_ECHO = "am_echo"
        AM_ERIC = "am_eric"
        AM_FENRIR = "am_fenrir"
        AM_LIAM = "am_liam"
        AM_MICHAEL = "am_michael"
        AM_ONYX = "am_onyx"
        AM_PUCK = "am_puck"
        AM_SANTA = "am_santa"


    prompt: str = Field(
        default=""
    )
    voice: Voice = Field(
        default=Voice.AF_HEART, description="Voice ID for the desired voice."
    )
    speed: float = Field(
        default=1, description="Speed of the generated audio. Default is 1.0."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "voice": self.voice.value,
            "speed": self.speed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kokoro",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class StableAudio(FALNode):
    """
    Stable Audio generates high-quality audio from text with consistent results.
    audio, generation, stable, music, text-to-audio

    Use cases:
    - Generate consistent audio
    - Create reliable soundtracks
    - Produce predictable audio
    - Generate stable music
    - Create dependable audio content
    """

    prompt: str = Field(
        default="", description="The prompt to generate audio from"
    )
    steps: int = Field(
        default=100, description="The number of steps to denoise the audio for"
    )
    seconds_total: int = Field(
        default=30, description="The duration of the audio clip to generate"
    )
    seconds_start: int = Field(
        default=0, description="The start point of the audio clip to generate"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "steps": self.steps,
            "seconds_total": self.seconds_total,
            "seconds_start": self.seconds_start,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-audio",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class XTTS(FALNode):
    """
    XTTS generates expressive speech with voice cloning capabilities.
    audio, tts, speech, voice-cloning, expressive, text-to-speech

    Use cases:
    - Clone and generate voices
    - Create personalized speech
    - Produce voice-matched content
    - Generate custom voice audio
    - Create voice replications
    """

    class Language(Enum):
        """
        The language to use for generation. Defaults to English.
        """
        ENGLISH = "English"
        SPANISH = "Spanish"
        FRENCH = "French"
        GERMAN = "German"
        ITALIAN = "Italian"
        PORTUGUESE = "Portuguese"
        POLISH = "Polish"
        TURKISH = "Turkish"
        RUSSIAN = "Russian"
        DUTCH = "Dutch"
        CZECH = "Czech"
        ARABIC = "Arabic"
        CHINESE = "Chinese"
        JAPANESE = "Japanese"
        HUNGARIAN = "Hungarian"
        KOREAN = "Korean"
        HINDI = "Hindi"


    prompt: str = Field(
        default="", description="The text prompt you would like to convert to speech."
    )
    repetition_penalty: float = Field(
        default=5, description="The repetition penalty to use for generation. Defaults to 5.0."
    )
    language: Language = Field(
        default=Language.ENGLISH, description="The language to use for generation. Defaults to English."
    )
    gpt_cond_len: int = Field(
        default=30, description="The length of the GPT conditioning. Defaults to 30."
    )
    gpt_cond_chunk_len: int = Field(
        default=4, description="The length of the GPT conditioning chunks. Defaults to 4."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the voice file to match"
    )
    temperature: float = Field(
        default=0.75, description="The temperature to use for generation. Higher is more creative. Defaults to 0.75."
    )
    sample_rate: int = Field(
        default=24000, description="The sample rate of the audio. Defaults to 24000."
    )
    max_ref_length: int = Field(
        default=60, description="The maximum length of the reference. Defaults to 60."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "repetition_penalty": self.repetition_penalty,
            "language": self.language.value,
            "gpt_cond_len": self.gpt_cond_len,
            "gpt_cond_chunk_len": self.gpt_cond_chunk_len,
            "audio_url": self.audio,
            "temperature": self.temperature,
            "sample_rate": self.sample_rate,
            "max_ref_length": self.max_ref_length,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/xtts",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MinimaxMusicV2(FALNode):
    """
    Minimax Music
    audio, generation, text-to-audio, tts, professional

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    prompt: str = Field(
        default="", description="A description of the music, specifying style, mood, and scenario. 10-300 characters."
    )
    lyrics_prompt: str = Field(
        default="", description="Lyrics of the song. Use n to separate lines. You may add structure tags like [Intro], [Verse], [Chorus], [Bridge], [Outro] to enhance the arrangement. 10-3000 characters."
    )
    audio_setting: str = Field(
        default="", description="Audio configuration settings"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "lyrics_prompt": self.lyrics_prompt,
            "audio_setting": self.audio_setting,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax-music/v2",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class BeatovenSoundEffectGeneration(FALNode):
    """
    Sound Effect Generation
    audio, generation, text-to-audio, tts

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    prompt: str = Field(
        default="", description="Describe the sound effect you want to generate"
    )
    duration: float = Field(
        default=5, description="Length of the generated sound effect in seconds"
    )
    refinement: int = Field(
        default=40, description="Refinement level - Higher values may improve quality but take longer"
    )
    seed: str = Field(
        default="", description="Random seed for reproducible results - leave empty for random generation"
    )
    negative_prompt: str = Field(
        default="", description="Describe the types of sounds you don't want to generate in the output, avoid double-negatives, compare with positive prompts"
    )
    creativity: float = Field(
        default=16, description="Creativity level - higher values allow more creative interpretation of the prompt"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration,
            "refinement": self.refinement,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "creativity": self.creativity,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="beatoven/sound-effect-generation",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class BeatovenMusicGeneration(FALNode):
    """
    Music Generation
    audio, generation, text-to-audio, tts

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    prompt: str = Field(
        default="", description="Describe the music you want to generate"
    )
    duration: float = Field(
        default=90, description="Length of the generated music in seconds"
    )
    refinement: int = Field(
        default=100, description="Refinement level - higher values may improve quality but take longer"
    )
    seed: str = Field(
        default="", description="Random seed for reproducible results - leave empty for random generation"
    )
    negative_prompt: str = Field(
        default="", description="Describe what you want to avoid in the music (instruments, styles, moods). Leave blank for none."
    )
    creativity: float = Field(
        default=16, description="Creativity level - higher values allow more creative interpretation of the prompt"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration,
            "refinement": self.refinement,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "creativity": self.creativity,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="beatoven/music-generation",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MinimaxMusicV15(FALNode):
    """
    MiniMax (Hailuo AI) Music v1.5
    audio, generation, text-to-audio, tts, professional

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    prompt: str = Field(
        default="", description="Lyrics, supports [intro][verse][chorus][bridge][outro] sections. 10-600 characters."
    )
    lyrics_prompt: str = Field(
        default="", description="Control music generation. 10-3000 characters."
    )
    audio_setting: str = Field(
        default="", description="Audio configuration settings"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "lyrics_prompt": self.lyrics_prompt,
            "audio_setting": self.audio_setting,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax-music/v1.5",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class StableAudio25TextToAudio(FALNode):
    """
    Stable Audio 2.5
    audio, generation, text-to-audio, tts

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    prompt: str = Field(
        default="", description="The prompt to generate audio from"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    seconds_total: int = Field(
        default=190, description="The duration of the audio clip to generate"
    )
    num_inference_steps: int = Field(
        default=8, description="The number of steps to denoise the audio for"
    )
    guidance_scale: int = Field(
        default=1, description="How strictly the diffusion process adheres to the prompt text (higher values make your audio closer to your prompt)."
    )
    seed: int = Field(
        default=0
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "sync_mode": self.sync_mode,
            "seconds_total": self.seconds_total,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-audio-25/text-to-audio",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class SonautoV2Inpaint(FALNode):
    """
    Sonauto V2
    audio, generation, text-to-audio, tts

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    class OutputFormat(Enum):
        FLAC = "flac"
        MP3 = "mp3"
        WAV = "wav"
        OGG = "ogg"
        M4A = "m4a"


    lyrics_prompt: str = Field(
        default="", description="The lyrics sung in the generated song. An empty string will generate an instrumental track."
    )
    tags: list[str] = Field(
        default=[], description="Tags/styles of the music to generate. You can view a list of all available tags at https://sonauto.ai/tag-explorer."
    )
    prompt_strength: float = Field(
        default=2, description="Controls how strongly your prompt influences the output. Greater values adhere more to the prompt but sound less natural. (This is CFG.)"
    )
    output_bit_rate: str = Field(
        default="", description="The bit rate to use for mp3 and m4a formats. Not available for other formats."
    )
    num_songs: int = Field(
        default=1, description="Generating 2 songs costs 1.5x the price of generating 1 song. Also, note that using the same seed may not result in identical songs if the number of songs generated is changed."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.WAV
    )
    selection_crop: bool = Field(
        default=False, description="Crop to the selected region"
    )
    sections: list[InpaintSection] = Field(
        default=[], description="List of sections to inpaint. Currently, only one section is supported so the list length must be 1."
    )
    balance_strength: float = Field(
        default=0.7, description="Greater means more natural vocals. Lower means sharper instrumentals. We recommend 0.7."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file to alter. Must be a valid publicly accessible URL."
    )
    seed: str = Field(
        default="", description="The seed to use for generation. Will pick a random seed if not provided. Repeating a request with identical parameters (must use lyrics and tags, not prompt) and the same seed will generate the same song."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "lyrics_prompt": self.lyrics_prompt,
            "tags": self.tags,
            "prompt_strength": self.prompt_strength,
            "output_bit_rate": self.output_bit_rate,
            "num_songs": self.num_songs,
            "output_format": self.output_format.value,
            "selection_crop": self.selection_crop,
            "sections": [item.model_dump(exclude={"type"}) for item in self.sections],
            "balance_strength": self.balance_strength,
            "audio_url": self.audio,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="sonauto/v2/inpaint",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class SonautoV2TextToMusic(FALNode):
    """
    Create full songs in any style
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    class OutputFormat(Enum):
        FLAC = "flac"
        MP3 = "mp3"
        WAV = "wav"
        OGG = "ogg"
        M4A = "m4a"


    prompt: str = Field(
        default="", description="A description of the track you want to generate. This prompt will be used to automatically generate the tags and lyrics unless you manually set them. For example, if you set prompt and tags, then the prompt will be used to generate only the lyrics."
    )
    lyrics_prompt: str = Field(
        default="", description="The lyrics sung in the generated song. An empty string will generate an instrumental track."
    )
    tags: str = Field(
        default="", description="Tags/styles of the music to generate. You can view a list of all available tags at https://sonauto.ai/tag-explorer."
    )
    prompt_strength: float = Field(
        default=2, description="Controls how strongly your prompt influences the output. Greater values adhere more to the prompt but sound less natural. (This is CFG.)"
    )
    output_bit_rate: str = Field(
        default="", description="The bit rate to use for mp3 and m4a formats. Not available for other formats."
    )
    num_songs: int = Field(
        default=1, description="Generating 2 songs costs 1.5x the price of generating 1 song. Also, note that using the same seed may not result in identical songs if the number of songs generated is changed."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.WAV
    )
    bpm: str = Field(
        default="auto", description="The beats per minute of the song. This can be set to an integer or the literal string \"auto\" to pick a suitable bpm based on the tags. Set bpm to null to not condition the model on bpm information."
    )
    balance_strength: float = Field(
        default=0.7, description="Greater means more natural vocals. Lower means sharper instrumentals. We recommend 0.7."
    )
    seed: str = Field(
        default="", description="The seed to use for generation. Will pick a random seed if not provided. Repeating a request with identical parameters (must use lyrics and tags, not prompt) and the same seed will generate the same song."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "lyrics_prompt": self.lyrics_prompt,
            "tags": self.tags,
            "prompt_strength": self.prompt_strength,
            "output_bit_rate": self.output_bit_rate,
            "num_songs": self.num_songs,
            "output_format": self.output_format.value,
            "bpm": self.bpm,
            "balance_strength": self.balance_strength,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="sonauto/v2/text-to-music",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class Lyria2(FALNode):
    """
    Lyria 2 is Google's latest music generation model, you can generate any type of music with this model.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    prompt: str = Field(
        default="", description="The text prompt describing the music you want to generate"
    )
    seed: int = Field(
        default=-1, description="A seed for deterministic generation. If provided, the model will attempt to produce the same audio given the same prompt and other parameters."
    )
    negative_prompt: str = Field(
        default="low quality", description="A description of what to exclude from the generated audio"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/lyria2",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class CassetteaiSoundEffectsGenerator(FALNode):
    """
    Create stunningly realistic sound effects in seconds - CassetteAI's Sound Effects Model generates high-quality SFX up to 30 seconds long in just 1 second of processing time
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    prompt: str = Field(
        default="", description="The prompt to generate SFX."
    )
    duration: int = Field(
        default=0, description="The duration of the generated SFX in seconds."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="cassetteai/sound-effects-generator",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class CassetteaiMusicGenerator(FALNode):
    """
    CassetteAI’s model generates a 30-second sample in under 2 seconds and a full 3-minute track in under 10 seconds. At 44.1 kHz stereo audio, expect a level of professional consistency with no breaks, no squeaks, and no random interruptions in your creations.  
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    prompt: str = Field(
        default="", description="The prompt to generate music from."
    )
    duration: int = Field(
        default=0, description="The duration of the generated music in seconds."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="cassetteai/music-generator",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class KokoroHindi(FALNode):
    """
    A fast and expressive Hindi text-to-speech model with clear pronunciation and accurate intonation.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    class Voice(Enum):
        """
        Voice ID for the desired voice.
        """
        HF_ALPHA = "hf_alpha"
        HF_BETA = "hf_beta"
        HM_OMEGA = "hm_omega"
        HM_PSI = "hm_psi"


    prompt: str = Field(
        default=""
    )
    voice: Voice = Field(
        default="", description="Voice ID for the desired voice."
    )
    speed: float = Field(
        default=1, description="Speed of the generated audio. Default is 1.0."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "voice": self.voice.value,
            "speed": self.speed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kokoro/hindi",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class KokoroBritishEnglish(FALNode):
    """
    A high-quality British English text-to-speech model offering natural and expressive voice synthesis.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    class Voice(Enum):
        """
        Voice ID for the desired voice.
        """
        BF_ALICE = "bf_alice"
        BF_EMMA = "bf_emma"
        BF_ISABELLA = "bf_isabella"
        BF_LILY = "bf_lily"
        BM_DANIEL = "bm_daniel"
        BM_FABLE = "bm_fable"
        BM_GEORGE = "bm_george"
        BM_LEWIS = "bm_lewis"


    prompt: str = Field(
        default=""
    )
    voice: Voice = Field(
        default="", description="Voice ID for the desired voice."
    )
    speed: float = Field(
        default=1, description="Speed of the generated audio. Default is 1.0."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "voice": self.voice.value,
            "speed": self.speed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kokoro/british-english",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class KokoroAmericanEnglish(FALNode):
    """
    Kokoro is a lightweight text-to-speech model that delivers comparable quality to larger models while being significantly faster and more cost-efficient.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    class Voice(Enum):
        """
        Voice ID for the desired voice.
        """
        AF_HEART = "af_heart"
        AF_ALLOY = "af_alloy"
        AF_AOEDE = "af_aoede"
        AF_BELLA = "af_bella"
        AF_JESSICA = "af_jessica"
        AF_KORE = "af_kore"
        AF_NICOLE = "af_nicole"
        AF_NOVA = "af_nova"
        AF_RIVER = "af_river"
        AF_SARAH = "af_sarah"
        AF_SKY = "af_sky"
        AM_ADAM = "am_adam"
        AM_ECHO = "am_echo"
        AM_ERIC = "am_eric"
        AM_FENRIR = "am_fenrir"
        AM_LIAM = "am_liam"
        AM_MICHAEL = "am_michael"
        AM_ONYX = "am_onyx"
        AM_PUCK = "am_puck"
        AM_SANTA = "am_santa"


    prompt: str = Field(
        default=""
    )
    voice: Voice = Field(
        default=Voice.AF_HEART, description="Voice ID for the desired voice."
    )
    speed: float = Field(
        default=1, description="Speed of the generated audio. Default is 1.0."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "voice": self.voice.value,
            "speed": self.speed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kokoro/american-english",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class Zonos(FALNode):
    """
    Clone voice of any person and speak anything in their voice using zonos' voice cloning.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    prompt: str = Field(
        default="", description="The content generated using cloned voice."
    )
    reference_audio: AudioRef = Field(
        default=AudioRef(), description="The reference audio."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "reference_audio_url": self.reference_audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/zonos",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class KokoroItalian(FALNode):
    """
    A high-quality Italian text-to-speech model delivering smooth and expressive speech synthesis.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    class Voice(Enum):
        """
        Voice ID for the desired voice.
        """
        IF_SARA = "if_sara"
        IM_NICOLA = "im_nicola"


    prompt: str = Field(
        default=""
    )
    voice: Voice = Field(
        default="", description="Voice ID for the desired voice."
    )
    speed: float = Field(
        default=1, description="Speed of the generated audio. Default is 1.0."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "voice": self.voice.value,
            "speed": self.speed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kokoro/italian",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class KokoroBrazilianPortuguese(FALNode):
    """
    A natural and expressive Brazilian Portuguese text-to-speech model optimized for clarity and fluency.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    class Voice(Enum):
        """
        Voice ID for the desired voice.
        """
        PF_DORA = "pf_dora"
        PM_ALEX = "pm_alex"
        PM_SANTA = "pm_santa"


    prompt: str = Field(
        default=""
    )
    voice: Voice = Field(
        default="", description="Voice ID for the desired voice."
    )
    speed: float = Field(
        default=1, description="Speed of the generated audio. Default is 1.0."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "voice": self.voice.value,
            "speed": self.speed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kokoro/brazilian-portuguese",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class KokoroFrench(FALNode):
    """
    An expressive and natural French text-to-speech model for both European and Canadian French.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    class Voice(Enum):
        """
        Voice ID for the desired voice.
        """
        FF_SIWIS = "ff_siwis"


    prompt: str = Field(
        default=""
    )
    voice: Voice = Field(
        default="", description="Voice ID for the desired voice."
    )
    speed: float = Field(
        default=1, description="Speed of the generated audio. Default is 1.0."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "voice": self.voice.value,
            "speed": self.speed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kokoro/french",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class KokoroJapanese(FALNode):
    """
    A fast and natural-sounding Japanese text-to-speech model optimized for smooth pronunciation.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    class Voice(Enum):
        """
        Voice ID for the desired voice.
        """
        JF_ALPHA = "jf_alpha"
        JF_GONGITSUNE = "jf_gongitsune"
        JF_NEZUMI = "jf_nezumi"
        JF_TEBUKURO = "jf_tebukuro"
        JM_KUMO = "jm_kumo"


    prompt: str = Field(
        default=""
    )
    voice: Voice = Field(
        default="", description="Voice ID for the desired voice."
    )
    speed: float = Field(
        default=1, description="Speed of the generated audio. Default is 1.0."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "voice": self.voice.value,
            "speed": self.speed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kokoro/japanese",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class KokoroMandarinChinese(FALNode):
    """
    A highly efficient Mandarin Chinese text-to-speech model that captures natural tones and prosody.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    class Voice(Enum):
        """
        Voice ID for the desired voice.
        """
        ZF_XIAOBEI = "zf_xiaobei"
        ZF_XIAONI = "zf_xiaoni"
        ZF_XIAOXIAO = "zf_xiaoxiao"
        ZF_XIAOYI = "zf_xiaoyi"
        ZM_YUNJIAN = "zm_yunjian"
        ZM_YUNXI = "zm_yunxi"
        ZM_YUNXIA = "zm_yunxia"
        ZM_YUNYANG = "zm_yunyang"


    prompt: str = Field(
        default=""
    )
    voice: Voice = Field(
        default="", description="Voice ID for the desired voice."
    )
    speed: float = Field(
        default=1, description="Speed of the generated audio. Default is 1.0."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "voice": self.voice.value,
            "speed": self.speed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kokoro/mandarin-chinese",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class KokoroSpanish(FALNode):
    """
    A natural-sounding Spanish text-to-speech model optimized for Latin American and European Spanish.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    class Voice(Enum):
        """
        Voice ID for the desired voice.
        """
        EF_DORA = "ef_dora"
        EM_ALEX = "em_alex"
        EM_SANTA = "em_santa"


    prompt: str = Field(
        default=""
    )
    voice: Voice = Field(
        default="", description="Voice ID for the desired voice."
    )
    speed: float = Field(
        default=1, description="Speed of the generated audio. Default is 1.0."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "voice": self.voice.value,
            "speed": self.speed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kokoro/spanish",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class Yue(FALNode):
    """
    YuE is a groundbreaking series of open-source foundation models designed for music generation, specifically for transforming lyrics into full songs.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    lyrics: str = Field(
        default="", description="The prompt to generate an image from. Must have two sections. Sections start with either [chorus] or a [verse]."
    )
    genres: str = Field(
        default="", description="The genres (separated by a space ' ') to guide the music generation."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "lyrics": self.lyrics,
            "genres": self.genres,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/yue",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MmaudioV2TextToAudio(FALNode):
    """
    MMAudio generates synchronized audio given text inputs. It can generate sounds described by a prompt.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    prompt: str = Field(
        default="", description="The prompt to generate the audio for."
    )
    num_steps: int = Field(
        default=25, description="The number of steps to generate the audio for."
    )
    duration: float = Field(
        default=8, description="The duration of the audio to generate."
    )
    cfg_strength: float = Field(
        default=4.5, description="The strength of Classifier Free Guidance."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator"
    )
    mask_away_clip: bool = Field(
        default=False, description="Whether to mask away the clip."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate the audio for."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "num_steps": self.num_steps,
            "duration": self.duration,
            "cfg_strength": self.cfg_strength,
            "seed": self.seed,
            "mask_away_clip": self.mask_away_clip,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/mmaudio-v2/text-to-audio",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class MinimaxMusic(FALNode):
    """
    Generate music from text prompts using the MiniMax model, which leverages advanced AI techniques to create high-quality, diverse musical compositions.
    audio, generation, text-to-audio, sound

    Use cases:
    - Sound effect generation
    - Music composition
    - Audio content creation
    - Background music generation
    - Podcast audio production
    """

    prompt: str = Field(
        default="", description="Lyrics with optional formatting. You can use a newline to separate each line of lyrics. You can use two newlines to add a pause between lines. You can use double hash marks (##) at the beginning and end of the lyrics to add accompaniment. Maximum 600 characters."
    )
    reference_audio: AudioRef = Field(
        default=AudioRef(), description="Reference song, should contain music and vocals. Must be a .wav or .mp3 file longer than 15 seconds."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "reference_audio_url": self.reference_audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax-music",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]