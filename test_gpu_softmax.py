import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import time

head_num = 32 # 32 for llama2-7b and 40 for llama2-13b
bs_list = [1,8,16,32] # batch_size
seql_list = [1,128,256,512,1024,2048,4096] # sequence length
for bs in bs_list:
    for sl in seql_list:
        x = torch.rand((bs, head_num , sl, 128)).cuda()
        COUNT = 100000
        # warmup
        for _ in range(1000):
            out = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)
            torch.cuda.synchronize()


        t0 = time.time()
        for _ in range(COUNT):
            out = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)
            torch.cuda.synchronize()
        t1 = time.time() - t0

        print(f'BS={bs} seql={sl} Time:{t1/COUNT*1000:.4f}ms')
