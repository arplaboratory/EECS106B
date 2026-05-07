# ----------------------------------------------------------------------------
# Copyright (c) 2026 C. K. Wolfe. All rights reserved.
#
# NOT FREE TO USE. Redistribution, modification, or commercial use of this
# file (and the bridge code in this package, excluding the vendored upstream
# dreamerv3 sources) is not permitted without explicit written permission
# from C. K. Wolfe.
# ----------------------------------------------------------------------------

"""omni_drones.learning.dreamerv3

PyTorch / IsaacSim bridge to the JAX dreamerv3 reference implementation,
with SkyDreamer-style modifications (informed decoder, smoothness regularizer,
multi-phase training schedule).
"""

from .policy import DreamerV3Policy

__all__ = ["DreamerV3Policy"]
