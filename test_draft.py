#!/usr/bin/env python3
"""Test with draft model - matches what Gallama does."""
import os, sys, time, gc
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,0,4,5'
import torch

from exllamav3.model.config import Config
from exllamav3.model.model import Model
from exllamav3.cache import Cache

# Main model
main_path = '/home/remichu/gallama/full_weight/Qwen3.6-27B-8.0bpw'
draft_path = '/home/remichu/gallama/models/Qwen3.6-27B-DFlash'

gpus = [94, 0, 0, 0, 0, 0]
use_per_device = [int(x * 1024**3) for x in gpus]

print('=== Loading main model ===', flush=True)
t0 = time.time()
main_config = Config.from_directory(main_path)
main_model = Model.from_config(main_config)
cache_size = 512000
main_cache = Cache(main_model, max_num_tokens=cache_size)
main_model.load(use_per_device=use_per_device, max_chunk_size=2048, progressbar=False)
free_mb, total_mb = torch.cuda.mem_get_info(0)
used_mb = (total_mb - free_mb) / 1024**2
print(f'Main model loaded: {time.time()-t0:.1f}s, VRAM used={used_mb:.0f}MB', flush=True)

# Draft model (like Gallama does)
if os.path.isdir(draft_path):
    print('\n=== Loading draft model (same GPU) ===', flush=True)
    t0 = time.time()
    draft_config = Config.from_directory(draft_path)
    draft_model = Model.from_config(draft_config)
    draft_cache = Cache(draft_model, max_num_tokens=cache_size)
    try:
        draft_model.load(use_per_device=use_per_device, max_chunk_size=2048, progressbar=False)
        free_mb, total_mb = torch.cuda.mem_get_info(0)
        used_mb2 = (total_mb - free_mb) / 1024**2
        print(f'Draft model loaded: {time.time()-t0:.1f}s, VRAM used={used_mb2:.0f}MB (+{used_mb2-used_mb:.0f}MB)', flush=True)
    except Exception as e:
        print(f'Draft FAILED: {e}', flush=True)
        import traceback
        traceback.print_exc()
else:
    print(f'\nDraft model not found at: {draft_path}', flush=True)

print('\nDone')
