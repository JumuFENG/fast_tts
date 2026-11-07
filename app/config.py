import tomllib
from pathlib import Path


CONFIG = tomllib.loads((Path(__file__).resolve().parent.parent / "config.toml").read_text())
