# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recording quality score of the clinical 10 m walk test with U-turn (body-worn IMUs).

Typical use::

    from package import assess_trial
    a = assess_trial("data/healthy/HS_1/HS_1_1", segmentation="segmentation.json")
    a.scores.global_score, a.scores.groups, a.scores.per_sensor
    a.table()                     # one row per indicator
"""

from .io import iter_trial_dirs, load_files, load_trial
from .params import default_params
from .pipeline import Assessment, assess, assess_files, assess_trial
from .timebase import align

__version__ = "0.2.0"
__all__ = ["Assessment", "align", "assess", "assess_files", "assess_trial", "default_params",
           "iter_trial_dirs", "load_files", "load_trial"]
