"""
traffic_violation.models.base
==============================
Structural typing contracts (Protocols) for all detection backends.

Using :class:`typing.Protocol` instead of ABC inheritance means that any
object whose methods match the interface is automatically compatible — no
explicit ``class Foo(DetectorProtocol)`` is required.  This makes the ONNX
and YOLO backends interchangeable without coupling them to a common base class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from traffic_violation.utils.geometry import Det


@runtime_checkable
class DetectorProtocol(Protocol):
    """Structural interface that every detection backend must satisfy.

    A conforming implementation must provide:

    * :meth:`predict` — run inference and return a list of :class:`~traffic_violation.utils.geometry.Det`.
    * :meth:`warmup` — pre-allocate GPU / JIT resources before the first real call.

    The ``@runtime_checkable`` decorator allows ``isinstance(obj, DetectorProtocol)``
    checks at runtime, which is useful in logging and testing.
    """

    def predict(
        self,
        img: np.ndarray,
        imgsz: int,
        conf: float,
        iou: float,
    ) -> list[Det]:
        """Run inference on *img* and return all bounding-box detections.

        Args:
            img:   BGR image array of shape ``(H, W, 3)``.
            imgsz: Longest-side resolution to which the image is resized
                   before inference (e.g. ``640``).
            conf:  Minimum detection confidence threshold in ``[0, 1]``.
            iou:   IoU threshold for the internal NMS pass inside the model.

        Returns:
            A list of :class:`~traffic_violation.utils.geometry.Det` objects
            in the original image's coordinate space.
        """
        ...

    def warmup(self) -> None:
        """Run a dummy inference pass to pre-allocate runtime resources.

        Call once after construction to avoid cold-start latency on the first
        real image.  Implementations should silently swallow all errors so
        that a warmup failure never breaks the pipeline.
        """
        ...
