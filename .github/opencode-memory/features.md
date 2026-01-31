# Features Log

This file tracks nodes and features added by the automated OpenCode agent.

## Format

Each entry should follow this format:
```
## YYYY-MM-DD - Feature/Node Name
- Endpoint: `provider/model/path`
- File: `src/nodetool/nodes/fal/<file>.py`
- Description: Brief description of what was added
```

---

## 2026-01-14 - New FAL Models Added

### ElevenLabs Scribe V2
- Endpoint: `fal-ai/elevenlabs/speech-to-text/scribe-v2`
- File: `src/nodetool/nodes/fal/speech_to_text.py`
- Description: Added ElevenLabs Scribe V2 speech-to-text model with improved accuracy, word-level timestamps, speaker diarization, and keyterms for biasing transcription.

### Nova SR
- Endpoint: `fal-ai/nova-sr`
- File: `src/nodetool/nodes/fal/text_to_audio.py`
- Description: Added Nova SR audio super-resolution model that enhances muffled 16kHz speech audio into crystal-clear 48kHz audio.

### LTX-2 19B Image to Video
- Endpoint: `fal-ai/ltx-2-19b/image-to-video`
- File: `src/nodetool/nodes/fal/image_to_video.py`
- Description: Added LTX-2 19B image-to-video model for generating high-quality videos with synchronized audio, camera motion control, and multi-scale generation support.

---

## Initial Setup - 2026-01-14

Repository configured with OpenCode memory and automated sync workflow.
