import torch
import time
import numpy as np
import argparse
import pandas as pd
import torch.cuda.profiler as profiler

# HOW TO RUN : nvprof --profile-from-start off --print-gpu-summary  python pytorch_benchmark_explore_v2.py

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def matrix_cal(shape):
    N=shape

    print("#" * 50)
    print(f"Mat Shape = {N}x{N}")
    print("#" * 50)

    x = torch.randn((N, N), requires_grad=True).to(device)
    y = torch.randn((N, N), requires_grad=True).to(device)

    #warmup
    for itr in range(1000):
        z = torch.matmul(x,y)

    #Using nvprof
    for itr in range(10):
        with torch.autograd.profiler.emit_nvtx() as pfvtx:
            profiler.start()
            z = torch.matmul(x,y)
        torch.cuda.synchronize()
        profiler.stop()


    cuda_event = []
    time_sync_api = []

    for itr in range(100):
        ## Using CUDA events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        z = torch.matmul(x,y)
        end.record()
        end.synchronize()
        cuda_event.append(start.elapsed_time(end))

        ## Using time API and Sync
        start = time.time()
        z = torch.matmul(x,y)
        torch.cuda.synchronize()
        end = time.time()
        time_sync_api.append((end-start)*1000)


    print(f"Using CUDA Events   = {np.array(cuda_event)[1:].mean()}ms")
    print(f"Using TIME SYNC API = {np.array(time_sync_api)[1:].mean()}ms")

    df = pd.DataFrame()
    df["cuda_event"] = cuda_event
    df["time_sync"] = time_sync_api
    df.to_csv("inference_benchmark_v2.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', default=16,
                        help="Set the batch size used in inference", type=int)

    args = parser.parse_args()
    matrix_cal(args.shape)
    print("\n\n\n")