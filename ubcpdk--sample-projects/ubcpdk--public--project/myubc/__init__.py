import pathlib

from gdsfactory.config import CONF

home = pathlib.Path.home()
cwd = pathlib.Path.cwd()
module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent

CONF.max_cellname_length = 64

__version__ = "0.0.0"
