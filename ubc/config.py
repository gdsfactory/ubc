"""
loads a configuration from `default_config` in this file (low priority)
which can be overwritten with config.yml found in the current working directory (high priority)
"""

__all__ = ["CONFIG", "conf"]

import io
import pathlib

from omegaconf import OmegaConf

default_config = io.StringIO(
    """
username: JoaquinMatres
"""
)

cwd = pathlib.Path.cwd()
cwd_config = cwd / "config.yml"
config_base = OmegaConf.load(default_config)
module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent

try:
    config_cwd = OmegaConf.load(cwd_config)
except Exception:
    config_cwd = OmegaConf.create()
conf = OmegaConf.merge(config_base, config_cwd)


class Config:
    module = module_path
    repo = repo_path


CONFIG = Config()


if __name__ == "__main__":
    print(CONFIG.repo)
