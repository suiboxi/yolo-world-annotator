"""Inference building blocks used by the annotator pipeline.

The package deliberately keeps image slicing, crop handling and box merging
independent from Qt and from any particular detector.  This makes the same
code usable by the GUI worker, benchmark scripts and unit tests.
"""

from yolo_world_annotator.inference.sahi_runner import (
    SAHIConfig,
    SAHIInferenceRunner,
    SAHIResult,
    generate_tiles,
)

__all__ = ["SAHIConfig", "SAHIInferenceRunner", "SAHIResult", "generate_tiles"]
