""" data analysis module
"""


from .chop import chop
from .find_bandwidth import find_bandwidth
from .read_mat import read_mat
from .remove_baseline import remove_baseline
from .windowed_mean import windowed_mean

__all__ = ["read_mat", "remove_baseline", "chop", "windowed_mean", "find_bandwidth"]
