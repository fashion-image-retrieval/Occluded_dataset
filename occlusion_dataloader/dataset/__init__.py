
from .import utils
from .base import BaseDataset



def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)
    