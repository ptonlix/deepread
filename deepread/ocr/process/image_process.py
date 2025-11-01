"""Image processing utilities for DeepSeek-OCR.

Adapted from DeepSeek-OCR-vllm repository (https://github.com/deepseek-ai/DeepSeek-OCR)
under MIT License.

Copyright (c) 2025 DeepSeek
"""

import math
from typing import Any, List, Tuple

import torch
import torchvision.transforms as T  # type: ignore[import-untyped]
from PIL import Image, ImageOps
from transformers import LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: set[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def count_tiles(
    orig_width: int,
    orig_height: int,
    min_num: int = 2,
    max_num: int = 6,
    image_size: int = 640,
) -> Tuple[int, int]:
    """Count tiles for image cropping."""
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios_set = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios_set, orig_width, orig_height, image_size
    )

    return target_aspect_ratio


def dynamic_preprocess(
    image: Image.Image, min_num: int = 2, max_num: int = 6, image_size: int = 640
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """Dynamically preprocess image by splitting into tiles."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios_set = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios_set, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    return processed_images, target_aspect_ratio


class ImageTransform:
    """Transform PIL image to tensor."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        self.mean = mean
        self.std = std
        self.normalize = normalize

        transform_pipelines: List[Any] = [T.ToTensor()]

        if normalize:
            transform_pipelines.append(T.Normalize(mean, std))

        self.transform = T.Compose(transform_pipelines)

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        result = self.transform(pil_img)
        assert isinstance(result, torch.Tensor)
        return result


class DeepseekOCRProcessor(ProcessorMixin):
    """Processor for DeepSeek-OCR images and text.

    Adapted to accept configuration parameters instead of using global config.
    """

    tokenizer_class: tuple[str, str] = ("LlamaTokenizer", "LlamaTokenizerFast")  # type: ignore[assignment]
    attributes: list[str] = ["tokenizer"]

    def __init__(
        self,
        tokenizer: LlamaTokenizerFast,
        image_size: int = 640,
        base_size: int = 1024,
        patch_size: int = 16,
        downsample_ratio: int = 4,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        pad_token: str = "<｜▁pad▁｜>",
        ignore_id: int = -100,
        **kwargs: Any,
    ) -> None:
        self.image_size = image_size
        self.base_size = base_size
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = normalize
        self.downsample_ratio = downsample_ratio

        self.image_transform = ImageTransform(
            mean=image_mean, std=image_std, normalize=normalize
        )

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"  # must set this for batch inference

        # add the pad_token as special token to use 'tokenizer.pad_token' and 'tokenizer.pad_token_id'
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": pad_token})

        image_token_id_raw = self.tokenizer.vocab.get(image_token)
        if image_token_id_raw is None:
            raise ValueError(
                f"Image token '{image_token}' not found in tokenizer vocabulary"
            )
        self.image_token_id: int = int(image_token_id_raw)

        self.image_token = image_token
        self.pad_token = pad_token
        self.ignore_id = ignore_id

        super().__init__(
            tokenizer,
            **kwargs,
        )

    @property
    def bos_id(self) -> int:
        result = self.tokenizer.bos_token_id
        assert isinstance(result, int)
        return result

    @property
    def eos_id(self) -> int:
        result = self.tokenizer.eos_token_id
        assert isinstance(result, int)
        return result

    @property
    def pad_id(self) -> int:
        result = self.tokenizer.pad_token_id
        assert isinstance(result, int)
        return result

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> List[int]:
        """Encode text to token ids."""
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        assert isinstance(encoded, list)
        t: List[int] = [int(token_id) for token_id in encoded]

        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]

        return t

    def decode(self, t: List[int], **kwargs: Any) -> str:
        """Decode token ids to text."""
        return self.tokenizer.decode(t, **kwargs)

    def tokenize_with_images(
        self,
        images: List[Image.Image],
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        bos: bool = True,
        eos: bool = True,
        cropping: bool = True,
        min_crops: int = 2,
        max_crops: int = 6,
    ) -> List:
        """Tokenize prompt with images, returning processed data for vLLM.

        Args:
            images: List of PIL Images to process
            prompt: Prompt string containing <image> tokens
            bos: Whether to add beginning of sequence token
            eos: Whether to add end of sequence token
            cropping: Whether to use cropping mode
            min_crops: Minimum number of crops
            max_crops: Maximum number of crops

        Returns:
            List containing [input_ids, pixel_values, images_crop, images_seq_mask,
                           images_spatial_crop, num_image_tokens, image_shapes]
        """
        assert prompt.count(self.image_token) == len(
            images
        ), f"Number of <image> tokens ({prompt.count(self.image_token)}) must match number of images ({len(images)})"

        text_splits = prompt.split(self.image_token)
        images_list, images_crop_list, images_seq_mask, images_spatial_crop = (
            [],
            [],
            [],
            [],
        )
        image_shapes = []
        num_image_tokens = []
        tokenized_str = []

        for text_sep, image in zip(text_splits, images):
            # encode text_sep
            tokenized_sep = self.encode(text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            image_shapes.append(image.size)

            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio: Tuple[int, int] = (1, 1)
                images_crop_raw: List[Image.Image] = []
            else:
                if cropping:
                    images_crop_raw, crop_ratio = dynamic_preprocess(
                        image,
                        min_num=min_crops,
                        max_num=max_crops,
                        image_size=self.image_size,
                    )
                else:
                    crop_ratio = (1, 1)
                    images_crop_raw = []

            # process the global view
            if self.image_size <= 640 and not cropping:
                image = image.resize((self.image_size, self.image_size))

            global_view = ImageOps.pad(
                image,
                (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            images_list.append(self.image_transform(global_view))

            # record height / width crop num
            num_width_tiles, num_height_tiles = crop_ratio
            images_spatial_crop.append([num_width_tiles, num_height_tiles])

            if num_width_tiles > 1 or num_height_tiles > 1:
                # process the local views
                for crop_img in images_crop_raw:
                    images_crop_list.append(self.image_transform(crop_img))

            # add image tokens
            num_queries = math.ceil(
                (self.image_size // self.patch_size) / self.downsample_ratio
            )
            num_queries_base = math.ceil(
                (self.base_size // self.patch_size) / self.downsample_ratio
            )

            # Build tokenized_image list
            base_tokens: List[int] = [self.image_token_id] * num_queries_base
            base_tokens.append(self.image_token_id)
            tokenized_image: List[int] = (base_tokens * num_queries_base).copy()
            tokenized_image.append(self.image_token_id)
            if num_width_tiles > 1 or num_height_tiles > 1:
                tile_tokens: List[int] = [self.image_token_id] * (
                    num_queries * num_width_tiles
                )
                tile_tokens.append(self.image_token_id)
                additional_tokens: List[int] = (
                    tile_tokens * (num_queries * num_height_tiles)
                ).copy()
                tokenized_image.extend(additional_tokens)
            tokenized_str.extend(tokenized_image)
            images_seq_mask += [True] * len(tokenized_image)
            num_image_tokens.append(len(tokenized_image))

        # process the last text split
        tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        # add the bos and eos tokens
        if bos:
            tokenized_str = [self.bos_id] + tokenized_str
            images_seq_mask = [False] + images_seq_mask
        if eos:
            tokenized_str = tokenized_str + [self.eos_id]
            images_seq_mask = images_seq_mask + [False]

        assert len(tokenized_str) == len(
            images_seq_mask
        ), f"tokenized_str length {len(tokenized_str)} != images_seq_mask length {len(images_seq_mask)}"

        masked_tokenized_str = []
        for token_index in tokenized_str:
            if token_index != self.image_token_id:
                masked_tokenized_str.append(token_index)
            else:
                masked_tokenized_str.append(self.ignore_id)

        assert (
            len(tokenized_str) == len(images_seq_mask) == len(masked_tokenized_str)
        ), (
            f"tokenized_str length {len(tokenized_str)}, masked_tokenized_str length {len(masked_tokenized_str)}, "
            f"images_seq_mask length {len(images_seq_mask)} are not equal"
        )

        input_ids: torch.LongTensor = torch.LongTensor(tokenized_str)
        target_ids: torch.LongTensor = torch.LongTensor(masked_tokenized_str)
        images_seq_mask_tensor: torch.Tensor = torch.tensor(
            images_seq_mask, dtype=torch.bool
        )

        # set input_ids < 0 | input_ids == self.image_token_id as ignore_id
        mask = (input_ids < 0) | (input_ids == self.image_token_id)
        target_ids[mask] = self.ignore_id
        input_ids[input_ids < 0] = self.pad_id

        # Remove the ending eos token for inference mode
        assert input_ids[-1].item() == self.eos_id
        input_ids_sliced = input_ids[:-1]
        images_seq_mask_sliced = images_seq_mask_tensor[:-1]

        if len(images_list) == 0:
            pixel_values = torch.zeros((1, 3, self.base_size, self.base_size))
            images_spatial_crop_tensor = torch.zeros((1, 1), dtype=torch.long)
            images_crop = torch.zeros(
                (1, 3, self.image_size, self.image_size)
            ).unsqueeze(0)
        else:
            pixel_values = torch.stack(images_list, dim=0)
            images_spatial_crop_tensor = torch.tensor(
                images_spatial_crop, dtype=torch.long
            )
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0).unsqueeze(0)
            else:
                images_crop = torch.zeros(
                    (1, 3, self.image_size, self.image_size)
                ).unsqueeze(0)

        input_ids_final = input_ids_sliced.unsqueeze(0)

        return [
            [
                input_ids_final,
                pixel_values,
                images_crop,
                images_seq_mask_sliced,
                images_spatial_crop_tensor,
                num_image_tokens,
                image_shapes,
            ]
        ]
