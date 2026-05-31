"""
traffic_violation
==================
Traffic violation detection package for two-wheelers.

Public API::

    from traffic_violation import TrafficViolationDetector

    model = TrafficViolationDetector("./models")
    result = model.predict("street_image.jpg")
    # result == {"violations": [{"num_riders": 2, "helmet_violations": 1, "license_plate": "AP07AB1234"}]}

Acceleration (ONNX Runtime)::

    from traffic_violation.accelerate import export_all
    export_all("./models")                     # exports all .pt → .onnx

    import os
    os.environ["TV_USE_ONNX"] = "1"
    model = TrafficViolationDetector("./models")  # now uses ONNX Runtime (3–4× faster)

Logging::

    import logging
    logging.basicConfig(level=logging.INFO)    # see timing and pipeline decisions
"""

import logging

from traffic_violation.pipeline import TrafficViolationDetector

# PEP 396 version string.
__version__ = "2.0.0"

__all__ = ["TrafficViolationDetector", "__version__"]

# Install a NullHandler so the library never emits log output unless the
# calling application configures a handler.
logging.getLogger(__name__).addHandler(logging.NullHandler())
