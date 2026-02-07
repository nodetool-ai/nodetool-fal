from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


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

    class OutputLoraFormat(Enum):
        """
        Dictates the naming scheme for the output weights
        """
        FAL = "fal"
        COMFY = "comfy"


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

    class OutputLoraFormat(Enum):
        """
        Dictates the naming scheme for the output weights
        """
        FAL = "fal"
        COMFY = "comfy"


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

    class OutputLoraFormat(Enum):
        """
        Dictates the naming scheme for the output weights
        """
        FAL = "fal"
        COMFY = "comfy"


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

    class OutputLoraFormat(Enum):
        """
        Dictates the naming scheme for the output weights
        """
        FAL = "fal"
        COMFY = "comfy"


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

    class OutputLoraFormat(Enum):
        """
        Dictates the naming scheme for the output weights
        """
        FAL = "fal"
        COMFY = "comfy"


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

    class OutputLoraFormat(Enum):
        """
        Dictates the naming scheme for the output weights
        """
        FAL = "fal"
        COMFY = "comfy"


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

class ZImageTrainer(FALNode):
    """
    Train LoRAs on Z-Image Turbo, a super fast text-to-image model of 6B parameters developed by Tongyi-MAI.
    training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    class TrainingType(Enum):
        """
        Type of training to perform. Use 'content' to focus on the content of the images, 'style' to focus on the style of the images, and 'balanced' to focus on a combination of both.
        """
        CONTENT = "content"
        STYLE = "style"
        BALANCED = "balanced"


    steps: int = Field(
        default=1000, description="Total number of training steps."
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to zip archive with images of a consistent style. Try to use at least 10 images, although more is better. The zip can also contain a text file for each image. The text file should be named: ROOT.txt For example: photo.txt This text file can be used to specify the edit instructions for the image pair. If no text file is provided, the default_caption will be used. If no default_caption is provided, the training will fail."
    )
    training_type: TrainingType = Field(
        default=TrainingType.BALANCED, description="Type of training to perform. Use 'content' to focus on the content of the images, 'style' to focus on the style of the images, and 'balanced' to focus on a combination of both."
    )
    learning_rate: float = Field(
        default=0.0001, description="Learning rate applied to trainable parameters."
    )
    default_caption: str = Field(
        default="", description="Default caption to use when caption files are missing. If None, missing captions will cause an error."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "training_type": self.training_type.value,
            "learning_rate": self.learning_rate,
            "default_caption": self.default_caption,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "training_type", "learning_rate", "default_caption"]

class Flux2TrainerEdit(FALNode):
    """
    Fine-tune FLUX.2 [dev] from Black Forest Labs with custom datasets. Create specialized LoRA adaptations for specific editing tasks.
    flux, training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    class OutputLoraFormat(Enum):
        """
        Dictates the naming scheme for the output weights
        """
        FAL = "fal"
        COMFY = "comfy"


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
            application="fal-ai/flux-2-trainer/edit",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption", "output_lora_format"]

class Flux2Trainer(FALNode):
    """
    Fine-tune FLUX.2 [dev] from Black Forest Labs with custom datasets. Create specialized LoRA adaptations for specific styles and domains.
    flux, training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    class OutputLoraFormat(Enum):
        """
        Dictates the naming scheme for the output weights
        """
        FAL = "fal"
        COMFY = "comfy"


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
            application="fal-ai/flux-2-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption", "output_lora_format"]

class QwenImageEditPlusTrainer(FALNode):
    """
    LoRA trainer for Qwen Image Edit Plus
    training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
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
            application="fal-ai/qwen-image-edit-plus-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption"]

class QwenImageEditTrainer(FALNode):
    """
    LoRA trainer for Qwen Image Edit
    training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    steps: int = Field(
        default=1000, description="Number of steps to train for"
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to the input data zip archive. The zip should contain pairs of images. The images should be named: ROOT_start.EXT and ROOT_end.EXT For example: photo_start.jpg and photo_end.jpg The zip can also contain a text file for each image pair. The text file should be named: ROOT.txt For example: photo.txt This text file can be used to specify the edit instructions for the image pair. If no text file is provided, the default_caption will be used. If no default_caption is provided, the training will fail."
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
            application="fal-ai/qwen-image-edit-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "default_caption"]

class QwenImageTrainer(FALNode):
    """
    Qwen Image LoRA training
    training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    steps: int = Field(
        default=1000, description="Total number of training steps to perform. Default is 4000."
    )
    image_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to zip archive with images for training. The archive should contain images and corresponding text files with captions. Each text file should have the same name as the image file it corresponds to (e.g., image1.jpg and image1.txt). If text files are missing for some images, you can provide a trigger_phrase to automatically create them. Supported image formats: PNG, JPG, JPEG, WEBP. Try to use at least 10 images, although more is better."
    )
    learning_rate: float = Field(
        default=0.0005, description="Learning rate for training. Default is 5e-4"
    )
    trigger_phrase: str = Field(
        default="", description="Default caption to use for images that don't have corresponding text files. If provided, missing .txt files will be created automatically."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_data_url_base64 = await context.image_to_base64(self.image_data_url)
        arguments = {
            "steps": self.steps,
            "image_data_url": f"data:image/png;base64,{image_data_url_base64}",
            "learning_rate": self.learning_rate,
            "trigger_phrase": self.trigger_phrase,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["steps", "image_data_url", "learning_rate", "trigger_phrase"]

class Wan22ImageTrainer(FALNode):
    """
    Wan 2.2 text to image LoRA trainer. Fine-tune Wan 2.2 for subjects and styles with unprecedented detail.
    training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    trigger_phrase: str = Field(
        default="", description="Trigger phrase for the model."
    )
    use_masks: bool = Field(
        default=True, description="Whether to use masks for the training data."
    )
    learning_rate: float = Field(
        default=0.0007, description="Learning rate for training."
    )
    use_face_cropping: bool = Field(
        default=False, description="Whether to use face cropping for the training data. When enabled, images will be cropped to the face before resizing."
    )
    training_data_url: str = Field(
        default="", description="URL to the training data."
    )
    steps: int = Field(
        default=1000, description="Number of training steps."
    )
    include_synthetic_captions: bool = Field(
        default=False, description="Whether to include synthetic captions."
    )
    is_style: bool = Field(
        default=False, description="Whether the training data is style data. If true, face specific options like masking and face detection will be disabled."
    )
    use_face_detection: bool = Field(
        default=True, description="Whether to use face detection for the training data. When enabled, images will use the center of the face as the center of the image when resizing."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "trigger_phrase": self.trigger_phrase,
            "use_masks": self.use_masks,
            "learning_rate": self.learning_rate,
            "use_face_cropping": self.use_face_cropping,
            "training_data_url": self.training_data_url,
            "steps": self.steps,
            "include_synthetic_captions": self.include_synthetic_captions,
            "is_style": self.is_style,
            "use_face_detection": self.use_face_detection,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-22-image-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["trigger_phrase", "use_masks", "learning_rate", "use_face_cropping", "training_data_url"]

class WanTrainerT2v(FALNode):
    """
    Train custom LoRAs for Wan-2.1 T2V 1.3B
    training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    number_of_steps: int = Field(
        default=400, description="The number of steps to train for."
    )
    training_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to zip archive with images of a consistent style. Try to use at least 10 images and/or videos, although more is better. In addition to images the archive can contain text files with captions. Each text file should have the same name as the image/video file it corresponds to."
    )
    trigger_phrase: str = Field(
        default="", description="The phrase that will trigger the model to generate an image."
    )
    learning_rate: float = Field(
        default=0.0002, description="The rate at which the model learns. Higher values can lead to faster training, but over-fitting."
    )
    auto_scale_input: bool = Field(
        default=False, description="If true, the input will be automatically scale the video to 81 frames at 16fps."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        training_data_url_base64 = await context.image_to_base64(self.training_data_url)
        arguments = {
            "number_of_steps": self.number_of_steps,
            "training_data_url": f"data:image/png;base64,{training_data_url_base64}",
            "trigger_phrase": self.trigger_phrase,
            "learning_rate": self.learning_rate,
            "auto_scale_input": self.auto_scale_input,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-trainer/t2v",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["number_of_steps", "training_data_url", "trigger_phrase", "learning_rate", "auto_scale_input"]

class WanTrainerT2v14b(FALNode):
    """
    Train custom LoRAs for Wan-2.1 T2V 14B
    training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    number_of_steps: int = Field(
        default=400, description="The number of steps to train for."
    )
    training_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to zip archive with images of a consistent style. Try to use at least 10 images and/or videos, although more is better. In addition to images the archive can contain text files with captions. Each text file should have the same name as the image/video file it corresponds to."
    )
    trigger_phrase: str = Field(
        default="", description="The phrase that will trigger the model to generate an image."
    )
    learning_rate: float = Field(
        default=0.0002, description="The rate at which the model learns. Higher values can lead to faster training, but over-fitting."
    )
    auto_scale_input: bool = Field(
        default=False, description="If true, the input will be automatically scale the video to 81 frames at 16fps."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        training_data_url_base64 = await context.image_to_base64(self.training_data_url)
        arguments = {
            "number_of_steps": self.number_of_steps,
            "training_data_url": f"data:image/png;base64,{training_data_url_base64}",
            "trigger_phrase": self.trigger_phrase,
            "learning_rate": self.learning_rate,
            "auto_scale_input": self.auto_scale_input,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-trainer/t2v-14b",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["number_of_steps", "training_data_url", "trigger_phrase", "learning_rate", "auto_scale_input"]

class WanTrainerI2v720p(FALNode):
    """
    Train custom LoRAs for Wan-2.1 I2V 720P
    training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    number_of_steps: int = Field(
        default=400, description="The number of steps to train for."
    )
    training_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to zip archive with images of a consistent style. Try to use at least 10 images and/or videos, although more is better. In addition to images the archive can contain text files with captions. Each text file should have the same name as the image/video file it corresponds to."
    )
    trigger_phrase: str = Field(
        default="", description="The phrase that will trigger the model to generate an image."
    )
    learning_rate: float = Field(
        default=0.0002, description="The rate at which the model learns. Higher values can lead to faster training, but over-fitting."
    )
    auto_scale_input: bool = Field(
        default=False, description="If true, the input will be automatically scale the video to 81 frames at 16fps."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        training_data_url_base64 = await context.image_to_base64(self.training_data_url)
        arguments = {
            "number_of_steps": self.number_of_steps,
            "training_data_url": f"data:image/png;base64,{training_data_url_base64}",
            "trigger_phrase": self.trigger_phrase,
            "learning_rate": self.learning_rate,
            "auto_scale_input": self.auto_scale_input,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-trainer/i2v-720p",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["number_of_steps", "training_data_url", "trigger_phrase", "learning_rate", "auto_scale_input"]

class WanTrainerFlf2v720p(FALNode):
    """
    Train custom LoRAs for Wan-2.1 FLF2V 720P
    training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    number_of_steps: int = Field(
        default=400, description="The number of steps to train for."
    )
    training_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to zip archive with images of a consistent style. Try to use at least 10 images and/or videos, although more is better. In addition to images the archive can contain text files with captions. Each text file should have the same name as the image/video file it corresponds to."
    )
    trigger_phrase: str = Field(
        default="", description="The phrase that will trigger the model to generate an image."
    )
    learning_rate: float = Field(
        default=0.0002, description="The rate at which the model learns. Higher values can lead to faster training, but over-fitting."
    )
    auto_scale_input: bool = Field(
        default=False, description="If true, the input will be automatically scale the video to 81 frames at 16fps."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        training_data_url_base64 = await context.image_to_base64(self.training_data_url)
        arguments = {
            "number_of_steps": self.number_of_steps,
            "training_data_url": f"data:image/png;base64,{training_data_url_base64}",
            "trigger_phrase": self.trigger_phrase,
            "learning_rate": self.learning_rate,
            "auto_scale_input": self.auto_scale_input,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-trainer/flf2v-720p",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["number_of_steps", "training_data_url", "trigger_phrase", "learning_rate", "auto_scale_input"]

class RecraftV3CreateStyle(FALNode):
    """
    Recraft V3 Create Style is capable of creating unique styles for Recraft V3 based on your images.
    training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    class BaseStyle(Enum):
        """
        The base style of the generated images, this topic is covered above.
        """
        ANY = "any"
        REALISTIC_IMAGE = "realistic_image"
        DIGITAL_ILLUSTRATION = "digital_illustration"
        VECTOR_ILLUSTRATION = "vector_illustration"
        REALISTIC_IMAGE_B_AND_W = "realistic_image/b_and_w"
        REALISTIC_IMAGE_HARD_FLASH = "realistic_image/hard_flash"
        REALISTIC_IMAGE_HDR = "realistic_image/hdr"
        REALISTIC_IMAGE_NATURAL_LIGHT = "realistic_image/natural_light"
        REALISTIC_IMAGE_STUDIO_PORTRAIT = "realistic_image/studio_portrait"
        REALISTIC_IMAGE_ENTERPRISE = "realistic_image/enterprise"
        REALISTIC_IMAGE_MOTION_BLUR = "realistic_image/motion_blur"
        REALISTIC_IMAGE_EVENING_LIGHT = "realistic_image/evening_light"
        REALISTIC_IMAGE_FADED_NOSTALGIA = "realistic_image/faded_nostalgia"
        REALISTIC_IMAGE_FOREST_LIFE = "realistic_image/forest_life"
        REALISTIC_IMAGE_MYSTIC_NATURALISM = "realistic_image/mystic_naturalism"
        REALISTIC_IMAGE_NATURAL_TONES = "realistic_image/natural_tones"
        REALISTIC_IMAGE_ORGANIC_CALM = "realistic_image/organic_calm"
        REALISTIC_IMAGE_REAL_LIFE_GLOW = "realistic_image/real_life_glow"
        REALISTIC_IMAGE_RETRO_REALISM = "realistic_image/retro_realism"
        REALISTIC_IMAGE_RETRO_SNAPSHOT = "realistic_image/retro_snapshot"
        REALISTIC_IMAGE_URBAN_DRAMA = "realistic_image/urban_drama"
        REALISTIC_IMAGE_VILLAGE_REALISM = "realistic_image/village_realism"
        REALISTIC_IMAGE_WARM_FOLK = "realistic_image/warm_folk"
        DIGITAL_ILLUSTRATION_PIXEL_ART = "digital_illustration/pixel_art"
        DIGITAL_ILLUSTRATION_HAND_DRAWN = "digital_illustration/hand_drawn"
        DIGITAL_ILLUSTRATION_GRAIN = "digital_illustration/grain"
        DIGITAL_ILLUSTRATION_INFANTILE_SKETCH = "digital_illustration/infantile_sketch"
        DIGITAL_ILLUSTRATION_2D_ART_POSTER = "digital_illustration/2d_art_poster"
        DIGITAL_ILLUSTRATION_HANDMADE_3D = "digital_illustration/handmade_3d"
        DIGITAL_ILLUSTRATION_HAND_DRAWN_OUTLINE = "digital_illustration/hand_drawn_outline"
        DIGITAL_ILLUSTRATION_ENGRAVING_COLOR = "digital_illustration/engraving_color"
        DIGITAL_ILLUSTRATION_2D_ART_POSTER_2 = "digital_illustration/2d_art_poster_2"
        DIGITAL_ILLUSTRATION_ANTIQUARIAN = "digital_illustration/antiquarian"
        DIGITAL_ILLUSTRATION_BOLD_FANTASY = "digital_illustration/bold_fantasy"
        DIGITAL_ILLUSTRATION_CHILD_BOOK = "digital_illustration/child_book"
        DIGITAL_ILLUSTRATION_CHILD_BOOKS = "digital_illustration/child_books"
        DIGITAL_ILLUSTRATION_COVER = "digital_illustration/cover"
        DIGITAL_ILLUSTRATION_CROSSHATCH = "digital_illustration/crosshatch"
        DIGITAL_ILLUSTRATION_DIGITAL_ENGRAVING = "digital_illustration/digital_engraving"
        DIGITAL_ILLUSTRATION_EXPRESSIONISM = "digital_illustration/expressionism"
        DIGITAL_ILLUSTRATION_FREEHAND_DETAILS = "digital_illustration/freehand_details"
        DIGITAL_ILLUSTRATION_GRAIN_20 = "digital_illustration/grain_20"
        DIGITAL_ILLUSTRATION_GRAPHIC_INTENSITY = "digital_illustration/graphic_intensity"
        DIGITAL_ILLUSTRATION_HARD_COMICS = "digital_illustration/hard_comics"
        DIGITAL_ILLUSTRATION_LONG_SHADOW = "digital_illustration/long_shadow"
        DIGITAL_ILLUSTRATION_MODERN_FOLK = "digital_illustration/modern_folk"
        DIGITAL_ILLUSTRATION_MULTICOLOR = "digital_illustration/multicolor"
        DIGITAL_ILLUSTRATION_NEON_CALM = "digital_illustration/neon_calm"
        DIGITAL_ILLUSTRATION_NOIR = "digital_illustration/noir"
        DIGITAL_ILLUSTRATION_NOSTALGIC_PASTEL = "digital_illustration/nostalgic_pastel"
        DIGITAL_ILLUSTRATION_OUTLINE_DETAILS = "digital_illustration/outline_details"
        DIGITAL_ILLUSTRATION_PASTEL_GRADIENT = "digital_illustration/pastel_gradient"
        DIGITAL_ILLUSTRATION_PASTEL_SKETCH = "digital_illustration/pastel_sketch"
        DIGITAL_ILLUSTRATION_POP_ART = "digital_illustration/pop_art"
        DIGITAL_ILLUSTRATION_POP_RENAISSANCE = "digital_illustration/pop_renaissance"
        DIGITAL_ILLUSTRATION_STREET_ART = "digital_illustration/street_art"
        DIGITAL_ILLUSTRATION_TABLET_SKETCH = "digital_illustration/tablet_sketch"
        DIGITAL_ILLUSTRATION_URBAN_GLOW = "digital_illustration/urban_glow"
        DIGITAL_ILLUSTRATION_URBAN_SKETCHING = "digital_illustration/urban_sketching"
        DIGITAL_ILLUSTRATION_VANILLA_DREAMS = "digital_illustration/vanilla_dreams"
        DIGITAL_ILLUSTRATION_YOUNG_ADULT_BOOK = "digital_illustration/young_adult_book"
        DIGITAL_ILLUSTRATION_YOUNG_ADULT_BOOK_2 = "digital_illustration/young_adult_book_2"
        VECTOR_ILLUSTRATION_BOLD_STROKE = "vector_illustration/bold_stroke"
        VECTOR_ILLUSTRATION_CHEMISTRY = "vector_illustration/chemistry"
        VECTOR_ILLUSTRATION_COLORED_STENCIL = "vector_illustration/colored_stencil"
        VECTOR_ILLUSTRATION_CONTOUR_POP_ART = "vector_illustration/contour_pop_art"
        VECTOR_ILLUSTRATION_COSMICS = "vector_illustration/cosmics"
        VECTOR_ILLUSTRATION_CUTOUT = "vector_illustration/cutout"
        VECTOR_ILLUSTRATION_DEPRESSIVE = "vector_illustration/depressive"
        VECTOR_ILLUSTRATION_EDITORIAL = "vector_illustration/editorial"
        VECTOR_ILLUSTRATION_EMOTIONAL_FLAT = "vector_illustration/emotional_flat"
        VECTOR_ILLUSTRATION_INFOGRAPHICAL = "vector_illustration/infographical"
        VECTOR_ILLUSTRATION_MARKER_OUTLINE = "vector_illustration/marker_outline"
        VECTOR_ILLUSTRATION_MOSAIC = "vector_illustration/mosaic"
        VECTOR_ILLUSTRATION_NAIVECTOR = "vector_illustration/naivector"
        VECTOR_ILLUSTRATION_ROUNDISH_FLAT = "vector_illustration/roundish_flat"
        VECTOR_ILLUSTRATION_SEGMENTED_COLORS = "vector_illustration/segmented_colors"
        VECTOR_ILLUSTRATION_SHARP_CONTRAST = "vector_illustration/sharp_contrast"
        VECTOR_ILLUSTRATION_THIN = "vector_illustration/thin"
        VECTOR_ILLUSTRATION_VECTOR_PHOTO = "vector_illustration/vector_photo"
        VECTOR_ILLUSTRATION_VIVID_SHAPES = "vector_illustration/vivid_shapes"
        VECTOR_ILLUSTRATION_ENGRAVING = "vector_illustration/engraving"
        VECTOR_ILLUSTRATION_LINE_ART = "vector_illustration/line_art"
        VECTOR_ILLUSTRATION_LINE_CIRCUIT = "vector_illustration/line_circuit"
        VECTOR_ILLUSTRATION_LINOCUT = "vector_illustration/linocut"


    images_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to zip archive with images, use PNG format. Maximum 5 images are allowed."
    )
    base_style: BaseStyle = Field(
        default=BaseStyle.DIGITAL_ILLUSTRATION, description="The base style of the generated images, this topic is covered above."
    )

    async def process(self, context: ProcessingContext) -> Any:
        images_data_url_base64 = await context.image_to_base64(self.images_data_url)
        arguments = {
            "images_data_url": f"data:image/png;base64,{images_data_url_base64}",
            "base_style": self.base_style.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/recraft/v3/create-style",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["images_data_url", "base_style"]

class TurboFluxTrainer(FALNode):
    """
    A blazing fast FLUX dev LoRA trainer for subjects and styles.
    flux, training, fine-tuning, lora, model-training, fast

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    class TrainingStyle(Enum):
        """
        Training style to use.
        """
        SUBJECT = "subject"
        STYLE = "style"


    images_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to zip archive with images of a consistent style. Try to use at least 10 images, although more is better."
    )
    trigger_phrase: str = Field(
        default="ohwx", description="Trigger phrase to be used in the captions. If None, a trigger word will not be used. If no captions are provide the trigger_work will be used instead of captions. If captions are provided, the trigger word will replace the `[trigger]` string in the captions."
    )
    steps: int = Field(
        default=1000, description="Number of steps to train the LoRA on."
    )
    learning_rate: float = Field(
        default=0.00115, description="Learning rate for the training."
    )
    training_style: TrainingStyle = Field(
        default=TrainingStyle.SUBJECT, description="Training style to use."
    )
    face_crop: bool = Field(
        default=True, description="Whether to try to detect the face and crop the images to the face."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        images_data_url_base64 = await context.image_to_base64(self.images_data_url)
        arguments = {
            "images_data_url": f"data:image/png;base64,{images_data_url_base64}",
            "trigger_phrase": self.trigger_phrase,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "training_style": self.training_style.value,
            "face_crop": self.face_crop,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/turbo-flux-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["images_data_url", "trigger_phrase", "steps", "learning_rate", "training_style"]

class WanTrainer(FALNode):
    """
    Train custom LoRAs for Wan-2.1 I2V 480P
    training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    number_of_steps: int = Field(
        default=400, description="The number of steps to train for."
    )
    training_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to zip archive with images of a consistent style. Try to use at least 10 images and/or videos, although more is better. In addition to images the archive can contain text files with captions. Each text file should have the same name as the image/video file it corresponds to."
    )
    trigger_phrase: str = Field(
        default="", description="The phrase that will trigger the model to generate an image."
    )
    learning_rate: float = Field(
        default=0.0002, description="The rate at which the model learns. Higher values can lead to faster training, but over-fitting."
    )
    auto_scale_input: bool = Field(
        default=False, description="If true, the input will be automatically scale the video to 81 frames at 16fps."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        training_data_url_base64 = await context.image_to_base64(self.training_data_url)
        arguments = {
            "number_of_steps": self.number_of_steps,
            "training_data_url": f"data:image/png;base64,{training_data_url_base64}",
            "trigger_phrase": self.trigger_phrase,
            "learning_rate": self.learning_rate,
            "auto_scale_input": self.auto_scale_input,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-trainer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["number_of_steps", "training_data_url", "trigger_phrase", "learning_rate", "auto_scale_input"]

class HunyuanVideoLoraTraining(FALNode):
    """
    Train Hunyuan Video lora on people, objects, characters and more!
    training, fine-tuning, lora, model-training

    Use cases:
    - Custom model fine-tuning
    - LoRA training for personalization
    - Style-specific model training
    - Brand-specific image generation
    - Specialized domain adaptation
    """

    trigger_word: str = Field(
        default="", description="The trigger word to use."
    )
    images_data_url: ImageRef = Field(
        default=ImageRef(), description="URL to zip archive with images. Try to use at least 4 images in general the more the better. In addition to images the archive can contain text files with captions. Each text file should have the same name as the image file it corresponds to."
    )
    steps: int = Field(
        default=0, description="Number of steps to train the LoRA on."
    )
    data_archive_format: str = Field(
        default="", description="The format of the archive. If not specified, the format will be inferred from the URL."
    )
    learning_rate: float = Field(
        default=0.0001, description="Learning rate to use for training."
    )
    do_caption: bool = Field(
        default=True, description="Whether to generate captions for the images."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        images_data_url_base64 = await context.image_to_base64(self.images_data_url)
        arguments = {
            "trigger_word": self.trigger_word,
            "images_data_url": f"data:image/png;base64,{images_data_url_base64}",
            "steps": self.steps,
            "data_archive_format": self.data_archive_format,
            "learning_rate": self.learning_rate,
            "do_caption": self.do_caption,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-video-lora-training",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["trigger_word", "images_data_url", "steps", "data_archive_format", "learning_rate"]