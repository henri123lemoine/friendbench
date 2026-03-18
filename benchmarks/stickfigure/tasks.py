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
from inspect_ai.tool import Tool, ToolError, tool

DATA_DIR = Path(__file__).resolve().parent / "data"
QUESTIONS_FILE = DATA_DIR / "questions.yaml"


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


def _active_images() -> dict[str, Path]:
    active = sample_active()
    if active is None:
        raise ToolError("No active sample is available.")

    metadata = active.sample.metadata or {}
    image_map = cast(dict[str, str] | None, metadata.get("images"))
    if not image_map:
        raise ToolError("This sample does not expose any images to the image tools.")

    return {image_id: DATA_DIR / rel_path for image_id, rel_path in image_map.items()}


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
        resolved_id, path = next(iter(images.items()))
        return resolved_id, path

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

        Returns image ids, filenames, and dimensions.
        Use these image ids with get_image_info and crop_image.
        """
        lines = ["Available images:"]
        for image_id, path in _active_images().items():
            with Image.open(path) as image:
                width, height = image.size
            lines.append(
                f"- {image_id}: {path.name} ({width}x{height} pixels)"
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
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or "unknown"
        return (
            f"Image '{resolved_id}' is file '{path.name}' with width={width}px, "
            f"height={height}px, format={image_format}."
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

        return [
            ContentText(
                text=(
                    f"Cropped image '{resolved_id}' returned for region "
                    f"x={x}, y={y}, width={right - x}, height={lower - y}."
                )
            ),
            ContentImage(image=_image_data_url(cropped), detail="high"),
        ]

    return execute


@task
def stickfigure():
    return Task(
        dataset=[
            Sample(
                input=[
                    ChatMessageUser(
                        content=[
                            ContentImage(
                                image=str(DATA_DIR / e["images"]),
                                detail="high",
                            ),
                            ContentText(text=e["input"]),
                        ]
                    ),
                ],
                target=e["target"],
                id=e["id"],
                metadata={"images": _normalize_images(e["images"])},
            )
            for e in yaml.safe_load(QUESTIONS_FILE.read_text()) or []
        ],
        solver=[use_tools(list_images(), get_image_info(), crop_image()), generate()],
        scorer=model_graded_qa(model="openai/gpt-5.4-mini"),
    )
