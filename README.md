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

### Dynamic Schema

- **FalAI** – call any fal.ai model by supplying a `llms.txt` URL, a fal.ai model URL, an endpoint id (e.g. `fal-ai/foo/bar`), or raw `llms.txt` content in the **model_info** field. The node fetches the OpenAPI schema from fal.ai, then **automatically fills all inputs and outputs** from that schema—you do not add inputs or outputs one by one.

  **How it works**

  - **model_info** accepts: a fal.ai llms.txt URL (e.g. `https://fal.ai/models/fal-ai/foo/llms.txt`), a model page URL (resolved to llms.txt), an endpoint id, or pasted llms.txt text.
  - Schema is resolved from the URL/content, then cached under `~/.cache/nodetool/fal_schema/` (or `NODETOOL_FAL_SCHEMA_CACHE`).
  - When the schema is available (from cache or after the first run), the node’s **dynamic inputs and outputs** are set from the OpenAPI schema so all parameters and result fields appear as slots.

  **If pasting a URL does nothing**

  - The first time you use a URL there is no cache, so the schema is only loaded when the node **runs**. After one successful run, reopening the workflow will show all inputs/outputs from cache.
  - To show inputs/outputs **before** running, the UI can call the async helper `resolve_dynamic_schema(model_info)` (from `nodetool.nodes.fal.dynamic_schema`) and apply the returned `dynamic_properties` and `dynamic_outputs` to the node. Backends can expose this as an API (e.g. when `model_info` changes) so pasting a URL immediately updates the node’s slots.

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

### Automated Model Sync

This repository includes an OpenCode workflow that periodically scans for new FAL endpoints and opens PRs with new nodes. See `.github/workflows/opencode-fal-model-sync.yml` for the schedule and prompt.
