"""Rendering backends. Importing this quiets library chatter that would
otherwise spill into the terminal UI (torch's macOS redirect notice, hub
progress bars, diffusers/transformers load warnings)."""

import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
