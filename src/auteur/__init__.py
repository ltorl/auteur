"""Auteur — an AI film director in your terminal."""

import logging as _logging
import os as _os

__version__ = "0.1.0"

# Quiet the ML stack before anything imports it: these libraries print straight
# to the terminal, which would scribble over the full-screen UI.
for _k, _v in {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error",
               "DIFFUSERS_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false"}.items():
    _os.environ.setdefault(_k, _v)
_logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(_logging.ERROR)
