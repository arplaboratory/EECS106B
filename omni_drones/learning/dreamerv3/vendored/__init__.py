# ----------------------------------------------------------------------------
# Copyright (c) 2026 C. K. Wolfe. All rights reserved (additions only).
#
# The contents of this `vendored/` directory are a copy of the JAX
# DreamerV3 reference implementation by Danijar Hafner, with small targeted
# patches by C. K. Wolfe (e.g. the SkyDreamer smoothness regularizer in
# vendored/dreamerv3/agent.py imag_loss). Upstream copyright and licensing
# applies to the unpatched portions; the patches themselves are the
# proprietary work of C. K. Wolfe and are NOT FREE TO USE without explicit
# written permission.
# ----------------------------------------------------------------------------

# Imported lazily by omni_drones.learning.dreamerv3.policy so the heavy JAX
# deps are only loaded when the dreamerv3 algorithm is actually selected.
