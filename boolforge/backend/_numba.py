#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import numba
    from numba import njit
    from numba.typed import List
    int64 = numba.int64
    __LOADED_NUMBA__ = True
except ModuleNotFoundError:
    # List = list
    # int64 = int
    # def njit(*args, **kwargs):
    #     def decorator(func):
    #         return func
    #     return decorator
    
    __LOADED_NUMBA__ = False
    
def _numba_required(feature: str):
    raise ImportError(
        f"{feature} requires numba. "
        "Install it with: pip install numba"
    )