from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class OutputLoraFormat(Enum):
    """
    Dictates the naming scheme for the output weights
    """
    FAL = "fal"
    COMFY = "comfy"


class ZImageBaseTrainer(FALNode):
    """
    Z-Image Trainer
    training, fine-tuning, lora, model-training

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    steps: int = Field(
        default=2000, description="Number of steps to train for"
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to the input data zip archive. The zip should contain pairs of images and corresponding captions. The images should be named: ROOT.EXT. For example: 001.jpg The corresponding captions should be named: ROOT.txt. For example: 001.txt If no text file is provided for an image, the default_caption will be used."
    )
    learning_rate: float = Field(
        default=0.0005, description="Learning rate."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image-base-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption"]

class ZImageTurboTrainerV2(FALNode):
    """
    Z Image Turbo Trainer V2
    training, fine-tuning, lora, model-training, fast

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    steps: int = Field(
        default=2000, description="Number of steps to train for"
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to the input data zip archive. The zip should contain pairs of images and corresponding captions. The images should be named: ROOT.EXT. For example: 001.jpg The corresponding captions should be named: ROOT.txt. For example: 001.txt If no text file is provided for an image, the default_caption will be used."
    )
    learning_rate: float = Field(
        default=0.0005, description="Learning rate."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image-turbo-trainer-v2",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption"]


class Flux2Klein9BBaseTrainerEdit(FALNode):
    """
    Flux 2 Klein 9B Base Trainer
    flux, training, fine-tuning, lora, model-training

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    steps: int = Field(
        default=1000, description="Total number of training steps."
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to the input data zip archive. The zip should contain pairs of images. The images should be named: ROOT_start.EXT and ROOT_end.EXT For example: photo_start.jpg and photo_end.jpg The zip can also contain up to four reference image for each image pair. The reference images should be named: ROOT_start.EXT, ROOT_start2.EXT, ROOT_start3.EXT, ROOT_start4.EXT, ROOT_end.EXT For example: photo_start.jpg, photo_start2.jpg, photo_end.jpg The zip can also contain a text file for each image pair. The text file should be named: ROOT.txt For example: photo.txt This text file can be used to specify the edit instructions for the image pair. If no text file is provided, the default_caption will be used. If no default_caption is provided, the training will fail."
    )
    learning_rate: float = Field(
        default=5e-05, description="Learning rate applied to trainable parameters."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )
    output_lora_format: OutputLoraFormat = Field(
        default=OutputLoraFormat.FAL, description="Dictates the naming scheme for the output weights"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
            "output_lora_format": self.output_lora_format.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-klein-9b-base-trainer/edit",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption", "output_lora_format"]


class Flux2Klein9BBaseTrainer(FALNode):
    """
    Flux 2 Klein 9B Base Trainer
    flux, training, fine-tuning, lora, model-training

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    steps: int = Field(
        default=1000, description="Total number of training steps."
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to zip archive with images of a consistent style. Try to use at least 10 images, although more is better. The zip can also contain a text file for each image. The text file should be named: ROOT.txt For example: photo.txt This text file can be used to specify the edit instructions for the image pair. If no text file is provided, the default_caption will be used. If no default_caption is provided, the training will fail."
    )
    learning_rate: float = Field(
        default=5e-05, description="Learning rate applied to trainable parameters."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )
    output_lora_format: OutputLoraFormat = Field(
        default=OutputLoraFormat.FAL, description="Dictates the naming scheme for the output weights"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
            "output_lora_format": self.output_lora_format.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-klein-9b-base-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption", "output_lora_format"]


class Flux2Klein4BBaseTrainer(FALNode):
    """
    Flux 2 Klein 4B Base Trainer
    flux, training, fine-tuning, lora, model-training

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    steps: int = Field(
        default=1000, description="Total number of training steps."
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to zip archive with images of a consistent style. Try to use at least 10 images, although more is better. The zip can also contain a text file for each image. The text file should be named: ROOT.txt For example: photo.txt This text file can be used to specify the edit instructions for the image pair. If no text file is provided, the default_caption will be used. If no default_caption is provided, the training will fail."
    )
    learning_rate: float = Field(
        default=5e-05, description="Learning rate applied to trainable parameters."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )
    output_lora_format: OutputLoraFormat = Field(
        default=OutputLoraFormat.FAL, description="Dictates the naming scheme for the output weights"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
            "output_lora_format": self.output_lora_format.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-klein-4b-base-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption", "output_lora_format"]


class Flux2Klein4BBaseTrainerEdit(FALNode):
    """
    Flux 2 Klein 4B Base Trainer
    flux, training, fine-tuning, lora, model-training

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    steps: int = Field(
        default=1000, description="Total number of training steps."
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to the input data zip archive. The zip should contain pairs of images. The images should be named: ROOT_start.EXT and ROOT_end.EXT For example: photo_start.jpg and photo_end.jpg The zip can also contain up to four reference image for each image pair. The reference images should be named: ROOT_start.EXT, ROOT_start2.EXT, ROOT_start3.EXT, ROOT_start4.EXT, ROOT_end.EXT For example: photo_start.jpg, photo_start2.jpg, photo_end.jpg The zip can also contain a text file for each image pair. The text file should be named: ROOT.txt For example: photo.txt This text file can be used to specify the edit instructions for the image pair. If no text file is provided, the default_caption will be used. If no default_caption is provided, the training will fail."
    )
    learning_rate: float = Field(
        default=5e-05, description="Learning rate applied to trainable parameters."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )
    output_lora_format: OutputLoraFormat = Field(
        default=OutputLoraFormat.FAL, description="Dictates the naming scheme for the output weights"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
            "output_lora_format": self.output_lora_format.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-klein-4b-base-trainer/edit",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption", "output_lora_format"]

class QwenImage2512TrainerV2(FALNode):
    """
    Qwen Image 2512 Trainer V2
    training, fine-tuning, lora, model-training

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    steps: int = Field(
        default=2000, description="Number of steps to train for"
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to the input data zip archive. The zip should contain pairs of images and corresponding captions. The images should be named: ROOT.EXT. For example: 001.jpg The corresponding captions should be named: ROOT.txt. For example: 001.txt If no text file is provided for an image, the default_caption will be used."
    )
    learning_rate: float = Field(
        default=0.0005, description="Learning rate."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-2512-trainer-v2",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption"]


class Flux2TrainerV2Edit(FALNode):
    """
    Flux 2 Trainer V2
    flux, training, fine-tuning, lora, model-training

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    steps: int = Field(
        default=1000, description="Total number of training steps."
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to the input data zip archive. The zip should contain pairs of images. The images should be named: ROOT_start.EXT and ROOT_end.EXT For example: photo_start.jpg and photo_end.jpg The zip can also contain up to four reference image for each image pair. The reference images should be named: ROOT_start.EXT, ROOT_start2.EXT, ROOT_start3.EXT, ROOT_start4.EXT, ROOT_end.EXT For example: photo_start.jpg, photo_start2.jpg, photo_end.jpg The zip can also contain a text file for each image pair. The text file should be named: ROOT.txt For example: photo.txt This text file can be used to specify the edit instructions for the image pair. If no text file is provided, the default_caption will be used. If no default_caption is provided, the training will fail."
    )
    learning_rate: float = Field(
        default=5e-05, description="Learning rate applied to trainable parameters."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )
    output_lora_format: OutputLoraFormat = Field(
        default=OutputLoraFormat.FAL, description="Dictates the naming scheme for the output weights"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
            "output_lora_format": self.output_lora_format.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-trainer-v2/edit",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption", "output_lora_format"]


class Flux2TrainerV2(FALNode):
    """
    Flux 2 Trainer V2
    flux, training, fine-tuning, lora, model-training

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    steps: int = Field(
        default=1000, description="Total number of training steps."
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to zip archive with images of a consistent style. Try to use at least 10 images, although more is better. The zip can also contain a text file for each image. The text file should be named: ROOT.txt For example: photo.txt This text file can be used to specify the edit instructions for the image pair. If no text file is provided, the default_caption will be used. If no default_caption is provided, the training will fail."
    )
    learning_rate: float = Field(
        default=5e-05, description="Learning rate applied to trainable parameters."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )
    output_lora_format: OutputLoraFormat = Field(
        default=OutputLoraFormat.FAL, description="Dictates the naming scheme for the output weights"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
            "output_lora_format": self.output_lora_format.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-trainer-v2",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption", "output_lora_format"]

class QwenImage2512Trainer(FALNode):
    """
    Qwen Image 2512 Trainer
    training, fine-tuning, lora, model-training

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    steps: int = Field(
        default=1000, description="Number of steps to train for"
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to the input data zip archive for text-to-image training. The zip should contain images with their corresponding text captions: image.EXT and image.txt For example: photo.jpg and photo.txt The text file contains the caption/prompt describing the target image. If no text file is provided for an image, the default_caption will be used. If no default_caption is provided and a text file is missing, the training will fail."
    )
    learning_rate: float = Field(
        default=0.0005, description="Learning rate for LoRA parameters."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-2512-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption"]

class QwenImageEdit2511Trainer(FALNode):
    """
    Qwen Image Edit 2511 Trainer
    training, fine-tuning, lora, model-training

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    steps: int = Field(
        default=1000, description="Number of steps to train for"
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to the input data zip archive. The zip should contain pairs of images. The images should be named: ROOT_start.EXT and ROOT_end.EXT For example: photo_start.jpg and photo_end.jpg The zip can also contain more than one reference image for each image pair. The reference images should be named: ROOT_start.EXT, ROOT_start2.EXT, ROOT_start3.EXT, ..., ROOT_end.EXT For example: photo_start.jpg, photo_start2.jpg, photo_end.jpg The Reference Image Count field should be set to the number of reference images. The zip can also contain a text file for each image pair. The text file should be named: ROOT.txt For example: photo.txt This text file can be used to specify the edit instructions for the image pair. If no text file is provided, the default_caption will be used. If no default_caption is provided, the training will fail."
    )
    learning_rate: float = Field(
        default=0.0001, description="Learning rate for LoRA parameters."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2511-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption"]

class QwenImageLayeredTrainer(FALNode):
    """
    Qwen Image Layered Trainer
    training, fine-tuning, lora, model-training

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    steps: int = Field(
        default=1000, description="Number of steps to train for"
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to the input data zip archive. The zip should contain groups of images. The images should be named: ROOT_start.EXT, ROOT_end.EXT, ROOT_end2.EXT, ..., ROOT_endN.EXT For example: photo_start.png, photo_end.png, photo_end2.png, ..., photo_endN.png The start image is the base image that will be decomposed into layers. The end images are the layers that will be added to the base image. ROOT_end.EXT is the first layer, ROOT_end2.EXT is the second layer, and so on. You can have up to 8 layers. All image groups must have the same number of output layers. The end images can contain transparent regions. Only PNG and WebP images are supported since these are the only formats that support transparency. The zip can also contain a text file for each image group. The text file should be named: ROOT.txt For example: photo.txt This text file can be used to specify a description of the base image. If no text file is provided, the default_caption will be used. If no default_caption is provided, the training will fail."
    )
    learning_rate: float = Field(
        default=0.0001, description="Learning rate for LoRA parameters."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-layered-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption"]

class QwenImageEdit2509Trainer(FALNode):
    """
    Qwen Image Edit 2509 Trainer
    training, fine-tuning, lora, model-training

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    steps: int = Field(
        default=1000, description="Number of steps to train for"
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to the input data zip archive. The zip should contain pairs of images. The images should be named: ROOT_start.EXT and ROOT_end.EXT For example: photo_start.jpg and photo_end.jpg The zip can also contain more than one reference image for each image pair. The reference images should be named: ROOT_start.EXT, ROOT_start2.EXT, ROOT_start3.EXT, ..., ROOT_end.EXT For example: photo_start.jpg, photo_start2.jpg, photo_end.jpg The Reference Image Count field should be set to the number of reference images. The zip can also contain a text file for each image pair. The text file should be named: ROOT.txt For example: photo.txt This text file can be used to specify the edit instructions for the image pair. If no text file is provided, the default_caption will be used. If no default_caption is provided, the training will fail."
    )
    learning_rate: float = Field(
        default=0.0001, description="Learning rate for LoRA parameters."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2509-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption"]