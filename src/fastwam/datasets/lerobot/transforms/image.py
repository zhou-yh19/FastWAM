import torch
import torch.nn as nn
import torchvision.transforms as TF
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


class ToTensor(nn.Module):
    """Convert uint8 image/video tensors from [0, 255] to float32 [0, 1]."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"ToTensor expects torch.Tensor, got {type(x).__name__}"
            )

        if x.dtype != torch.uint8:
            raise TypeError(
                f"ToTensor expects uint8 input, got {x.dtype}"
            )

        return x.to(dtype=torch.float32) / 255.0


class Pad(nn.Module):
    """Pad a [T, C, H, W] tensor."""

    def __init__(
        self,
        padding,
        fill=0,
        padding_mode: str = "constant",
    ):
        super().__init__()

        self.padding = tuple(padding)
        self.fill = fill
        self.padding_mode = padding_mode

        self.pad = TF.Pad(
            padding=self.padding,
            fill=self.fill,
            padding_mode=self.padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Pad expects [T, C, H, W], got shape {tuple(x.shape)}"
            )

        return self.pad(x)


class Letterbox(nn.Module):
    """Resize while preserving aspect ratio, then pad to a fixed canvas.

    Expected input:
        [T, C, H, W]

    Output:
        [T, C, target_h, target_w]

    `fill` is interpreted in the current tensor value range. When Letterbox
    follows ToTensor, use values in [0, 1].
    """

    def __init__(
        self,
        size,
        fill: float = 0.0,
        antialias: bool = True,
    ):
        super().__init__()

        if len(size) != 2:
            raise ValueError(
                f"`size` must be [target_h, target_w], got {size}"
            )

        self.target_h = int(size[0])
        self.target_w = int(size[1])
        self.fill = float(fill)
        self.antialias = bool(antialias)

        if self.target_h <= 0 or self.target_w <= 0:
            raise ValueError(
                "Letterbox target dimensions must be positive, "
                f"got {(self.target_h, self.target_w)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Letterbox expects torch.Tensor, got {type(x).__name__}"
            )

        if x.ndim != 4:
            raise ValueError(
                "Letterbox expects a [T, C, H, W] tensor, "
                f"got shape {tuple(x.shape)}"
            )

        if not x.is_floating_point():
            raise TypeError(
                "Letterbox expects floating-point input. "
                "Apply ToTensor before Letterbox. "
                f"Received dtype={x.dtype}"
            )

        src_h, src_w = x.shape[-2:]

        if src_h <= 0 or src_w <= 0:
            raise ValueError(
                f"Invalid source image size: H={src_h}, W={src_w}"
            )

        # Largest scale that keeps the complete image inside the target canvas.
        scale = min(
            self.target_h / src_h,
            self.target_w / src_w,
        )

        new_h = int(round(src_h * scale))
        new_w = int(round(src_w * scale))

        # Protect against floating-point rounding at the target boundary.
        new_h = max(1, min(self.target_h, new_h))
        new_w = max(1, min(self.target_w, new_w))

        x = F.resize(
            x,
            size=[new_h, new_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=self.antialias,
        )

        pad_h = self.target_h - new_h
        pad_w = self.target_w - new_w

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # torchvision padding order:
        # [left, top, right, bottom]
        x = F.pad(
            x,
            padding=[
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
            ],
            fill=self.fill,
            padding_mode="constant",
        )

        actual_size = tuple(x.shape[-2:])
        expected_size = (self.target_h, self.target_w)

        if actual_size != expected_size:
            raise RuntimeError(
                f"Letterbox produced spatial size {actual_size}, "
                f"expected {expected_size}"
            )

        return x


def compose_ta2_mosaic(
    cameras: list[torch.Tensor],
    canvas_size=(352, 256),
    head_tile=(256, 256),
    wrist_tile=(80, 128),
    fill: float = 0.5,
    antialias: bool = True,
) -> torch.Tensor:
    """Resize each camera once into its tile, then paste onto a fixed canvas.

    Expected camera order (matches the TA2 mono shape_meta):
        [head_camera, left_color, right_color]

    Each input tensor:
        [T, C, H, W] float in [0, 1]

    Output:
        [T, C, canvas_h, canvas_w]

    Layout (content 336x256, bottom gray pad to canvas_h):
        ┌──────────── head (256x256) ────────────┐
        ├──── left (80x128) ──┬── right (80x128) ─┤
        └────────────────────────────────────────┘

    Defaults are the monocular (left-eye) geometry. The TA2 cameras are side-by-side
    stereo, and feeding both eyes spent 43.8% of the canvas on a duplicate viewpoint
    without adding single-eye detail. Cropping to the left eye keeps every eye pixel
    (head 512x256 -> 256x256, wrist 256x80 -> 128x80 are exact halves) and drops
    tokens/frame from 192 to 88.

    canvas_size must be a multiple of 32 in both dims: the VAE downsamples by 16
    (`WanVideoVAE38.upsampling_factor`) and the DiT then patchifies the latent by 2
    (`patch_size=[1, 2, 2]`), so 16 alone is not enough.
    """
    if len(cameras) != 3:
        raise ValueError(
            f"compose_ta2_mosaic expects 3 cameras [head, left, right], got {len(cameras)}"
        )

    for i, cam in enumerate(cameras):
        if not isinstance(cam, torch.Tensor):
            raise TypeError(f"camera[{i}] must be a Tensor, got {type(cam).__name__}")
        if cam.ndim != 4:
            raise ValueError(
                f"camera[{i}] expects [T, C, H, W], got shape {tuple(cam.shape)}"
            )
        if not cam.is_floating_point():
            raise TypeError(
                f"camera[{i}] expects floating-point (apply ToTensor first), got {cam.dtype}"
            )

    head_lb = Letterbox(size=list(head_tile), fill=fill, antialias=antialias)
    wrist_lb = Letterbox(size=list(wrist_tile), fill=fill, antialias=antialias)

    head = head_lb(cameras[0])
    left = wrist_lb(cameras[1])
    right = wrist_lb(cameras[2])

    canvas_h, canvas_w = int(canvas_size[0]), int(canvas_size[1])
    head_h, head_w = int(head_tile[0]), int(head_tile[1])
    wrist_h, wrist_w = int(wrist_tile[0]), int(wrist_tile[1])

    if head_w != 2 * wrist_w:
        raise ValueError(
            f"TA2 mosaic expects head_w == 2*wrist_w for side-by-side wrists, "
            f"got head_w={head_w}, wrist_w={wrist_w}"
        )
    if head_w > canvas_w or (head_h + wrist_h) > canvas_h:
        raise ValueError(
            f"Tiles do not fit on canvas {canvas_size}: "
            f"head={head_tile}, wrist={wrist_tile}"
        )

    t, c = head.shape[0], head.shape[1]
    out = head.new_full((t, c, canvas_h, canvas_w), float(fill))

    x0 = (canvas_w - head_w) // 2
    out[:, :, 0:head_h, x0 : x0 + head_w] = head
    out[:, :, head_h : head_h + wrist_h, x0 : x0 + wrist_w] = left
    out[:, :, head_h : head_h + wrist_h, x0 + wrist_w : x0 + 2 * wrist_w] = right
    return out
