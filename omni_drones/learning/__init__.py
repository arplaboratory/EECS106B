# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from .ppo import *
from .happo import HAPPOPolicy
from .sac import SACPolicy
from .td3 import TD3Policy

# DreamerV3 (C. K. Wolfe, 2026) — additive registration only; PPO and the
# other existing algos above are unchanged.
try:
    from .dreamerv3 import DreamerV3Policy
except Exception as _dreamer_import_err:  # noqa: BLE001
    DreamerV3Policy = None
    import logging as _logging
    _logging.debug("DreamerV3Policy unavailable: %s", _dreamer_import_err)

ALGOS = {
    "mappo": MAPPOPolicy,
    "happo": HAPPOPolicy,
    "ppo": PPOPolicy,
    "ppo_rnn": PPORNNPolicy,
    "ppo_adapt": PPOAdaptivePolicy,
    "sac": SACPolicy,
    "td3": TD3Policy,
}
if DreamerV3Policy is not None:
    ALGOS["dreamerv3"] = DreamerV3Policy
