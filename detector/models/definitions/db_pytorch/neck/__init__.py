from .extractor import Extractor

__all__ = ['get_neck']

def get_neck(**kwargs):
    return eval('Extractor')(**kwargs)

