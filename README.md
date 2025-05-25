# Nodetool FAL Nodes

This repository packages a collection of nodes for [Nodetool](https://github.com/nodetool-ai/nodetool) that integrate the [FAL](https://fal.ai/) generative AI APIs. The nodes build on top of [nodetool-core](https://github.com/nodetool-ai/nodetool-core) and expose FAL models for use inside Nodetool workflows.

These nodes cover text, image, audio, and video generation along with speech transcription. They can be combined with other Nodetool packages to create complete multimodal pipelines.

## Installation

```bash
pip install git+https://github.com/nodetool-ai/nodetool-core.git
pip install fal-client
pip install git+https://github.com/nodetool-ai/nodetool-fal.git
```

Alternatively use `poetry` to install from `pyproject.toml`.

### Authentication

All nodes expect a valid `FAL_API_KEY` in the workflow environment. The key is exported to the underlying `fal-client` library when a node executes.

## Node Overview

The package defines nodes grouped by modality. Each node is a subclass of `FALNode` which provides the common logic for calling a FAL endpoint.

### Large Language Model

- **AnyLLM** – interface to multiple LLMs (Claude 3, Gemini, Llama, GPT‑4 and others) allowing you to select the model via the `model` field.

### Text to Image

- **IdeogramV2** and **IdeogramV2Turbo** – generate images with typography and creative control.
- **FluxV1Pro** and **FluxV1ProUltra** – high‑fidelity versions of the FLUX model series.
- **RecraftV3** – produces vector style images and brand assets.
- **Switti** – fast text‑to‑image transformer.
- **AuraFlowV03**, **FluxDev**, **FluxLora**, **FluxLoraInpainting**, **FluxSchnell**, **FluxSubject**, **FluxV1ProNew**, **SanaV1**, **OmniGenV1**, **StableDiffusionV35Large**, **Recraft20B**, **BriaV1**, **BriaV1Fast**, **BriaV1HD**, **FluxGeneral**, **StableDiffusionV3Medium**, **FastSDXL**, **FluxLoraTTI**, **StableCascade**, **LumaPhoton**, **LumaPhotonFlash**, **FastTurboDiffusion**, **FastLCMDiffusion**, **FastLightningSDXL**, **HyperSDXL**, **PlaygroundV25**, **LCMDiffusion**, **Fooocus**, **IllusionDiffusion**, **FastSDXLControlNetCanny**, **FluxDevImageToImage**, and **DiffusionEdge** – various FAL models for specialized generation tasks.

### Image to Image

- **FluxSchnellRedux**, **FluxDevRedux**, **FluxProRedux**, **FluxProUltraRedux**, **FluxProFill**, **FluxProCanny**, **FluxProDepth**, **FluxLoraCanny**, **FluxLoraDepth**, **IdeogramV2Edit**, **IdeogramV2Remix**, **BriaEraser**, **BriaProductShot**, **BriaBackgroundReplace**, **BriaGenFill**, **BriaExpand**, **BriaBackgroundRemove** – transform or edit existing images using the corresponding FAL models.

### Image to Video

- **HaiperImageToVideo**, **LumaDreamMachine**, **KlingVideo**, **KlingVideoPro**, **KlingVideoV2** – convert an image into a short video clip with control over duration and style.
- **CogVideoX**, **MiniMaxVideo**, **MiniMaxHailuo02**, **LTXVideo**, **StableVideo**, **FastSVD**, **AMTInterpolation**, **SadTalker**, **MuseTalk** – additional models for motion synthesis, interpolation and talking‑face generation.

### Text to Video

- **KlingTextToVideoV2** – create short video clips directly from text prompts.

### Text to Audio

- **MMAudioV2** – synthesise audio clips from text prompts.
- **StableAudio** – open source text‑to‑audio model.
- **F5TTS** – high quality text‑to‑speech with optional voice cloning.

### Speech to Text

- **Whisper** – transcribe or translate speech from audio files with optional diarization support.

## Metadata

Package metadata is stored in [`src/nodetool/package_metadata/nodetool-fal.json`](src/nodetool/package_metadata/nodetool-fal.json) which describes all nodes and their inputs/outputs for the Nodetool ecosystem.

## License

The project is distributed under the terms of the GNU Affero General Public License 3.0 as found in [`LICENSE.txt`](LICENSE.txt).

## Contributing

Issues and pull requests are welcome. See the documentation in the main [Nodetool](https://github.com/nodetool-ai/nodetool) repository for guidelines on developing new nodes and workflows.
