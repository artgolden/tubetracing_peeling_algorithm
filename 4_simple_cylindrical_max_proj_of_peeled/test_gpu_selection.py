import os
import cupy as cp

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Current GPU:", cp.cuda.Device().id)