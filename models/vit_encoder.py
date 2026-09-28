import logging
import os

import torch
from torch.hub import HASH_REGEX, download_url_to_file, urlparse

from dinov2.models import vision_transformer as vision_transformer_dinov2

_logger = logging.getLogger(__name__)

_WEIGHTS_DIR = "backbones/weights"
os.makedirs(_WEIGHTS_DIR, exist_ok=True)

_CKPT_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
)


def load():
    model = vision_transformer_dinov2.__dict__['vit_base'](
        patch_size=14,
        img_size=518,
        block_chunks=0,
        init_values=1e-8,
        num_register_tokens=4,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    ckpt_pth = download_cached_file(_CKPT_URL)
    state_dict = torch.load(ckpt_pth, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    return model


def download_cached_file(url, check_hash=True, progress=True):
    """
    Mostly copy-paste from timm library.
    (https://github.com/rwightman/pytorch-image-models/blob/29fda20e6d428bf636090ab207bbcf60617570ca/timm/models/_hub.py#L54)
    """
    if isinstance(url, (list, tuple)):
        url, filename = url
    else:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
    cached_file = os.path.join(_WEIGHTS_DIR, filename)
    if not os.path.exists(cached_file):
        _logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file
