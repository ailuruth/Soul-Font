"""
DMFont
Copyright (c) 2020-present NAVER Corp.
"""
import torch.nn as nn


def split_dim(x, dim, n_chunks):
    shape = x.shape
    assert shape[dim] % n_chunks == 0
    return x.view(*shape[:dim], n_chunks, shape[dim] // n_chunks, *shape[dim+1:])


# `weights_init` was removed here.
#
# Upstream DM-Font adopted it from NVlabs/FUNIT, which is licensed CC BY-NC-SA 4.0
# and so forbids commercial use — the one carve-out from DM-Font's own MIT grant.
# Soul Font is now sold, so it cannot ship. It was reachable only from
# dmfont/train.py; inference never touched it. Do not restore it from upstream.
# See NOTICE.


def spectral_norm(module):
    """ init & apply spectral norm """
    nn.init.xavier_uniform_(module.weight, 2 ** 0.5)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    return nn.utils.spectral_norm(module)
