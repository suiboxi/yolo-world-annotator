"""Runtime device selection shared by the GUI and model adapters."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re

import torch


class DeviceSelectionError(ValueError):
    """Raised when a requested inference device cannot be selected."""


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """A validated device selection suitable for Ultralytics inference."""

    requested: str
    torch_device: str
    description: str
    use_half: bool


_CUDA_PATTERN = re.compile(r"cuda(?::(?P<index>\d+))?\Z")


def resolve_device(requested: str | None = None) -> DeviceInfo:
    """Resolve ``auto``, ``cpu`` or ``cuda[:N]`` into a validated device."""

    value = (requested or os.environ.get("YOLO_WORLD_DEVICE", "auto")).strip().lower()
    if not value:
        value = "auto"

    if value == "cpu":
        return DeviceInfo(
            requested="cpu",
            torch_device="cpu",
            description="CPU / float32",
            use_half=False,
        )

    if value == "auto":
        if not torch.cuda.is_available():
            return DeviceInfo(
                requested="auto",
                torch_device="cpu",
                description="CPU / float32",
                use_half=False,
            )
        return _cuda_device(requested="auto", index=0)

    match = _CUDA_PATTERN.fullmatch(value)
    if match is None:
        raise DeviceSelectionError(
            f"无法识别设备 {value!r}；请使用 auto、cpu、cuda 或 cuda:N。"
        )
    if not torch.cuda.is_available():
        raise DeviceSelectionError(
            f"请求了 {value}，但当前 PyTorch CUDA 运行时不可用。"
        )
    index = int(match.group("index") or 0)
    return _cuda_device(requested=value, index=index)


def _cuda_device(*, requested: str, index: int) -> DeviceInfo:
    count = int(torch.cuda.device_count())
    if index >= count:
        raise DeviceSelectionError(
            f"请求了 cuda:{index}，但当前只检测到 {count} 个 CUDA 设备。"
        )
    name = str(torch.cuda.get_device_name(index))
    total_memory = torch.cuda.get_device_properties(index).total_memory / 1024**3
    return DeviceInfo(
        requested=requested,
        torch_device=f"cuda:{index}",
        description=f"{name} / cuda:{index} / {total_memory:.1f} GiB / float16",
        use_half=True,
    )


__all__ = ["DeviceInfo", "DeviceSelectionError", "resolve_device"]
