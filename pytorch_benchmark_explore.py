import torch
import time
import numpy as np
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
x = torch.randn((50, 50), requires_grad=True).to(device)
y = torch.randn((50, 50), requires_grad=True).to(device)

#warmup
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    z = torch.matmul(x,y)
torch.cuda.synchronize()

torch_profile = []
cuda_event = []
time_api = []
time_sync_api = []

for itr in range(1000):
    ## Using CUDA Profile
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        z = torch.matmul(x,y)
    torch.cuda.synchronize() # wait for previous GPU operations but anyway wait time is not considered for calc
    gpu_time = sum(evt.self_cuda_time_total for evt in prof.function_events)
    torch_profile.append(gpu_time / 1000)


    ## Using CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    z = torch.matmul(x,y)
    end.record()
    torch.cuda.synchronize()
    cuda_event.append(start.elapsed_time(end))


    ## Using time API
    start = time.time()
    z = torch.matmul(x,y)
    end = time.time()
    time_api.append((end-start)*1000)
    torch.cuda.synchronize() # wait for previous GPU operations but dont consider wait time for calc


    ## Using time API and Sync
    start = time.time()
    z = torch.matmul(x,y)
    torch.cuda.synchronize()
    end = time.time()
    time_sync_api.append((end-start)*1000)


print(f"Using TORCH Profile = {np.array(torch_profile).mean()}ms")
print(f"Using CUDA Events   = {np.array(cuda_event).mean()}ms")
print(f"Using TIME API      = {np.array(time_api).mean()}ms")
print(f"Using TIME SYNC API = {np.array(time_sync_api).mean()}ms")

df = pd.DataFrame()
df["torch_profile"] = torch_profile
df["cuda_event"] = cuda_event
df["time_api"] = time_api
df["time_sync"] = time_sync_api
df.to_csv("inference_benchmark.csv", index=False)

## RESULTS ##
# cuda:0
# Using TORCH Profile = 0.0522901119158268ms
# Using CUDA Events   = 0.03761523292819038ms
# Using TIME API      = 0.018018245697021484ms
# Using TIME SYNC API = 0.022334575653076172ms

