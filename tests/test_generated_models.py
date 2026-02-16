import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio

from nodetool.fal.generated_models import (
    IMAGE_MODELS,
    VIDEO_MODELS,
    TTS_MODELS,
    ASR_MODELS,
    MODEL_3D_MODELS,
)
from nodetool.metadata.types import (
    ImageModel,
    VideoModel,
    TTSModel,
    ASRModel,
    Model3DModel,
    Provider,
)
from nodetool.fal.fal_provider import FalProvider


class TestGeneratedModelLists:
    """Tests for generated model lists from all_models.json."""

    def test_image_models_not_empty(self):
        """IMAGE_MODELS should contain models."""
        assert len(IMAGE_MODELS) > 0

    def test_video_models_not_empty(self):
        """VIDEO_MODELS should contain models."""
        assert len(VIDEO_MODELS) > 0

    def test_tts_models_not_empty(self):
        """TTS_MODELS should contain models."""
        assert len(TTS_MODELS) > 0

    def test_asr_models_not_empty(self):
        """ASR_MODELS should contain models."""
        assert len(ASR_MODELS) > 0

    def test_3d_models_not_empty(self):
        """MODEL_3D_MODELS should contain models."""
        assert len(MODEL_3D_MODELS) > 0

    def test_image_models_type(self):
        """All items in IMAGE_MODELS should be ImageModel instances."""
        for m in IMAGE_MODELS:
            assert isinstance(m, ImageModel)

    def test_video_models_type(self):
        """All items in VIDEO_MODELS should be VideoModel instances."""
        for m in VIDEO_MODELS:
            assert isinstance(m, VideoModel)

    def test_tts_models_type(self):
        """All items in TTS_MODELS should be TTSModel instances."""
        for m in TTS_MODELS:
            assert isinstance(m, TTSModel)

    def test_asr_models_type(self):
        """All items in ASR_MODELS should be ASRModel instances."""
        for m in ASR_MODELS:
            assert isinstance(m, ASRModel)

    def test_3d_models_type(self):
        """All items in MODEL_3D_MODELS should be Model3DModel instances."""
        for m in MODEL_3D_MODELS:
            assert isinstance(m, Model3DModel)

    def test_all_models_have_fal_provider(self):
        """All generated models should use Provider.FalAI."""
        all_models = (
            list(IMAGE_MODELS)
            + list(VIDEO_MODELS)
            + list(TTS_MODELS)
            + list(ASR_MODELS)
            + list(MODEL_3D_MODELS)
        )
        for m in all_models:
            assert m.provider == Provider.FalAI

    def test_all_models_have_id(self):
        """All generated models should have a non-empty id."""
        all_models = (
            list(IMAGE_MODELS)
            + list(VIDEO_MODELS)
            + list(TTS_MODELS)
            + list(ASR_MODELS)
            + list(MODEL_3D_MODELS)
        )
        for m in all_models:
            assert m.id, f"Model {m.name} has empty id"

    def test_all_models_have_name(self):
        """All generated models should have a non-empty name."""
        all_models = (
            list(IMAGE_MODELS)
            + list(VIDEO_MODELS)
            + list(TTS_MODELS)
            + list(ASR_MODELS)
            + list(MODEL_3D_MODELS)
        )
        for m in all_models:
            assert m.name, f"Model {m.id} has empty name"

    def test_image_models_unique_ids(self):
        """IMAGE_MODELS should have unique model ids."""
        ids = [m.id for m in IMAGE_MODELS]
        assert len(ids) == len(set(ids))

    def test_video_models_unique_ids(self):
        """VIDEO_MODELS should have unique model ids."""
        ids = [m.id for m in VIDEO_MODELS]
        assert len(ids) == len(set(ids))

    def test_image_models_have_supported_tasks(self):
        """Image models should have supported_tasks populated."""
        for m in IMAGE_MODELS:
            assert len(m.supported_tasks) > 0
            for task in m.supported_tasks:
                assert task in ("text-to-image", "image-to-image")

    def test_video_models_have_supported_tasks(self):
        """Video models should have supported_tasks populated."""
        for m in VIDEO_MODELS:
            assert len(m.supported_tasks) > 0
            for task in m.supported_tasks:
                assert task in ("text-to-video", "image-to-video", "video-to-video")


class TestProviderGeneratedModels:
    """Tests that FalProvider returns generated model lists."""

    def test_get_available_image_models(self):
        """Provider should return all generated image models."""
        os.environ["FAL_API_KEY"] = "test-key"
        provider = FalProvider(secrets={"FAL_API_KEY": "test-key"})
        models = asyncio.run(provider.get_available_image_models())
        assert len(models) == len(IMAGE_MODELS)
        assert all(isinstance(m, ImageModel) for m in models)

    def test_get_available_video_models(self):
        """Provider should return all generated video models."""
        os.environ["FAL_API_KEY"] = "test-key"
        provider = FalProvider(secrets={"FAL_API_KEY": "test-key"})
        models = asyncio.run(provider.get_available_video_models())
        assert len(models) == len(VIDEO_MODELS)
        assert all(isinstance(m, VideoModel) for m in models)

    def test_get_available_tts_models(self):
        """Provider should return all generated TTS models."""
        os.environ["FAL_API_KEY"] = "test-key"
        provider = FalProvider(secrets={"FAL_API_KEY": "test-key"})
        models = asyncio.run(provider.get_available_tts_models())
        assert len(models) == len(TTS_MODELS)
        assert all(isinstance(m, TTSModel) for m in models)

    def test_get_available_asr_models(self):
        """Provider should return all generated ASR models."""
        os.environ["FAL_API_KEY"] = "test-key"
        provider = FalProvider(secrets={"FAL_API_KEY": "test-key"})
        models = asyncio.run(provider.get_available_asr_models())
        assert len(models) == len(ASR_MODELS)
        assert all(isinstance(m, ASRModel) for m in models)

    def test_get_available_3d_models(self):
        """Provider should return all generated 3D models."""
        os.environ["FAL_API_KEY"] = "test-key"
        provider = FalProvider(secrets={"FAL_API_KEY": "test-key"})
        models = asyncio.run(provider.get_available_3d_models())
        assert len(models) == len(MODEL_3D_MODELS)
        assert all(isinstance(m, Model3DModel) for m in models)

    def test_model_lists_are_copies(self):
        """Provider should return copies, not the original lists."""
        os.environ["FAL_API_KEY"] = "test-key"
        provider = FalProvider(secrets={"FAL_API_KEY": "test-key"})
        models1 = asyncio.run(provider.get_available_image_models())
        models2 = asyncio.run(provider.get_available_image_models())
        assert models1 is not models2
        assert models1 is not IMAGE_MODELS
