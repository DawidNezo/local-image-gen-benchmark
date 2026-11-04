import os
from pathlib import Path

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(Path.cwd() / ".hf_cache")

from imagebench.cli import main  # noqa: E402  (HF_HOME must be set before importing diffusers)

main()
