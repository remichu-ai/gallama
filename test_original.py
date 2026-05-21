#!/usr/bin/env python3
"""Test original (unpatched) exllamav3 loading qwen3.6-27B."""
import os, sys, time, gc
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,0,4,5'
import torch

from exllamav3.model.config import Config
from exllamav3.model.model import Model
from exllamav3.cache import Cache

model_path = '/home/remichu/gallama/full_weight/Qwen3.6-27B-8.0bpw'
gpus = [94, 0, 0, 0, 0, 0]
use_per_device = [int(x * 1024**3) for x in gpus]

for cache_size in [128000, 256000, 512000]:
    print(f'\n=== Testing cache_size={cache_size} ===', flush=True)
    
    config = Config.from_directory(model_path)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=cache_size)
    
    t0 = time.time()
    try:
        model.load(use_per_device=use_per_device, max_chunk_size=2048, max_output_size=32, progressbar=False)
        free_mb, total_mb = torch.cuda.mem_get_info(0)
        used_mb = (total_mb - free_mb) / 1024**2
        print(f'SUCCESS ({time.time()-t0:.1f}s) - VRAM used={used_mb:.0f}MB', flush=True)
    except Exception as e:
        try:
            free_mb, total_mb = torch.cuda.mem_get_info(0)
            used_mb = (total_mb - free_mb) / 1024**2
        except:
            used_mb = -1
        print(f'FAILED: {e}', flush=True)
        import traceback
        traceback.print_exc()
    
    try:
        model.unload()
    except:
        pass
    del model, cache, config
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print('\nDone')
