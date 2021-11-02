from .head import Head

__all__ = ['get_head']

def get_head(**kwargs):
    return eval('Head')(**kwargs)