import base64
from io import BytesIO
from pathlib import Path
from typing import cast

import yaml
from PIL import Image
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.log._samples import sample_active
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, ToolError, code_execution, tool

DATA_DIR = Path(__file__).resolve().parent / "data"
QUESTIONS_FILE = DATA_DIR / "questions.yaml"
CROP_TARGET_MAX_DIMENSION = 2048


def _image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _normalize_images(images: str | list[str] | dict[str, str]) -> dict[str, str]:
    if isinstance(images, str):
        return {"primary": images}
    if isinstance(images, list):
        if len(images) == 1:
            return {"primary": images[0]}
        return {f"image_{index + 1}": path for index, path in enumerate(images)}
    return images


def _sandbox_paths(images: dict[str, str]) -> dict[str, str]:
    sandbox_paths: dict[str, str] = {}
    used_paths: set[str] = set()

    for image_id, rel_path in images.items():
        filename = Path(rel_path).name
        sandbox_path = f"images/{filename}"
        if sandbox_path in used_paths:
            sandbox_path = f"images/{image_id}-{filename}"
        used_paths.add(sandbox_path)
        sandbox_paths[image_id] = sandbox_path

    return sandbox_paths


def _sample_from_entry(entry: dict[str, str]) -> Sample:
    image_map = _normalize_images(entry["images"])
    sandbox_map = _sandbox_paths(image_map)
    primary_image = next(iter(image_map.values()))

    return Sample(
        input=[
            ChatMessageUser(
                content=[
                    ContentImage(
                        image=str(DATA_DIR / primary_image),
                        detail="high",
                    ),
                    ContentText(text=entry["input"]),
                ]
            ),
        ],
        target=entry["target"],
        id=entry["id"],
        metadata={
            "images": image_map,
            "sandbox_images": sandbox_map,
        },
        files={
            sandbox_path: str(DATA_DIR / image_map[image_id])
            for image_id, sandbox_path in sandbox_map.items()
        },
    )


def _active_metadata_field(key: str) -> dict[str, str]:
    active = sample_active()
    if active is None:
        raise ToolError("No active sample is available.")
    value = cast(dict[str, str] | None, (active.sample.metadata or {}).get(key))
    if not value:
        raise ToolError(f"No {key.replace('_', ' ')} available for the image tools.")
    return value


def _active_images() -> dict[str, Path]:
    return {k: DATA_DIR / v for k, v in _active_metadata_field("images").items()}


def _active_sandbox_paths() -> dict[str, str]:
    return _active_metadata_field("sandbox_images")


def _resolve_image(image_id: str | None) -> tuple[str, Path]:
    images = _active_images()
    if image_id:
        path = images.get(image_id)
        if path is None:
            available = ", ".join(sorted(images))
            raise ToolError(
                f"Unknown image_id '{image_id}'. Available image ids: {available}."
            )
        return image_id, path

    if len(images) == 1:
        return next(iter(images.items()))

    available = ", ".join(sorted(images))
    raise ToolError(
        "image_id is required when multiple images are available. "
        f"Available image ids: {available}."
    )


@tool(parallel=False)
def list_images() -> Tool:
    async def execute() -> str:
        """
        List the images available for the current sample.

        Returns image ids, filenames, dimensions, and local code_execution paths.
        Use these image ids with get_image_info and crop_image.
        """
        lines = ["Available images:"]
        sandbox_paths = _active_sandbox_paths()
        for image_id, path in _active_images().items():
            with Image.open(path) as image:
                width, height = image.size
            lines.append(
                f"- {image_id}: {path.name} ({width}x{height} pixels, "
                f"code_execution path: {sandbox_paths[image_id]})"
            )
        return "\n".join(lines)

    return execute


@tool(parallel=False)
def get_image_info() -> Tool:
    async def execute(image_id: str = "") -> str:
        """
        Return dimensions and basic metadata for an available sample image.

        If image_id is omitted and the sample has only one image, that image is used.
        The origin for crop coordinates is the top-left corner.

        Args:
          image_id: Optional image id from list_images().
        """
        resolved_id, path = _resolve_image(image_id or None)
        sandbox_paths = _active_sandbox_paths()
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or "unknown"
        return (
            f"Image '{resolved_id}' is file '{path.name}' with width={width}px, "
            f"height={height}px, format={image_format}, and is available in "
            f"code_execution at '{sandbox_paths[resolved_id]}'."
        )

    return execute


@tool(parallel=False)
def crop_image() -> Tool:
    async def execute(
        x: int,
        y: int,
        width: int,
        height: int,
        image_id: str = "",
    ) -> list[ContentText | ContentImage]:
        """
        Crop a rectangular region from an available sample image and return it as a new image.

        Coordinates are pixel values measured from the top-left corner.
        Small crops are automatically enlarged before being returned, so each
        crop acts like a zoomed-in view of the selected region.
        If image_id is omitted and the sample has only one image, that image is used.

        Args:
          x: Left coordinate of the crop box in pixels.
          y: Top coordinate of the crop box in pixels.
          width: Width of the crop box in pixels.
          height: Height of the crop box in pixels.
          image_id: Optional image id from list_images().
        """
        if width <= 0 or height <= 0:
            raise ToolError("width and height must be positive integers.")

        resolved_id, path = _resolve_image(image_id or None)
        with Image.open(path) as image:
            image_width, image_height = image.size

            if x < 0 or y < 0:
                raise ToolError("x and y must be non-negative.")
            if x >= image_width or y >= image_height:
                raise ToolError("x and y must fall within the image bounds.")

            right = min(image_width, x + width)
            lower = min(image_height, y + height)
            if right <= x or lower <= y:
                raise ToolError("The requested crop region is empty.")

            cropped = image.crop((x, y, right, lower))
            crop_width = right - x
            crop_height = lower - y
            crop_max_dimension = max(crop_width, crop_height)
            if crop_max_dimension < CROP_TARGET_MAX_DIMENSION:
                scale = CROP_TARGET_MAX_DIMENSION / crop_max_dimension
                cropped = cropped.resize(
                    (
                        max(1, round(crop_width * scale)),
                        max(1, round(crop_height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )

        return [
            ContentText(
                text=(
                    f"Cropped image '{resolved_id}' returned for region "
                    f"x={x}, y={y}, width={crop_width}, height={crop_height}. "
                    f"Returned image size is {cropped.width}x{cropped.height} pixels."
                )
            ),
            ContentImage(image=_image_data_url(cropped), detail="high"),
        ]

    return execute


@task
def stickfigure():
    return Task(
        dataset=[_sample_from_entry(e) for e in yaml.safe_load(QUESTIONS_FILE.read_text()) or []],
        solver=[
            use_tools(
                list_images(),
                get_image_info(),
                crop_image(),
                code_execution(
                    providers={
                        "openai": False,
                        "python": {"timeout": 60},
                    }
                ),
            ),
            generate(),
        ],
        sandbox="local",
        scorer=model_graded_qa(model="openai/gpt-5.4-mini"),
    )
