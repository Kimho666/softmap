import math 

def saturation(x, M=8):
    x_max = 2**(M-1) - 1
    x_min = (-1) * (2**(M-1))

    max_mask = (x > x_max).int()
    x = x * (1-max_mask) + x_max * max_mask

    min_mask = (x < x_min).int()
    x = x * (1-min_mask) + x_min * min_mask 
    return x

def i_poly(q,S,M):
    a = 0.3585
    b = 1.353
    c = 0.344
    # q_b = torch.floor(b / S).to(torch.int8)
    # q_c = torch.floor(c / (a * (S**2))).to(torch.int16)
    q_b = saturation(torch.floor(b / S).to(torch.int8), M)
    q_c = saturation(torch.floor(c / (a * (S**2))).to(torch.int16), 2*M)
    S_out = a * (S**2)
    q_out = ((q + q_b).to(torch.int16))**2 + q_c
    q_out = saturation(((q + q_b).to(torch.int16))**2 + q_c, 2*M+5)
    # print(f'S:{S}, q:{q}, q_b:{q_b}, q_c:{q_c}, q_out:{q_out}, S_out:{S_out}')

    return q_out, S_out

def i_exp(q,S,M):

    # q_ln2 = torch.floor(math.log(2) / S).int()
    # q_ln2 = torch.floor(math.log(2) / S).to(torch.int8)
    q_ln2 = saturation(torch.floor(math.log(2) / S).to(torch.int8), 4)
    z = torch.floor(-q / q_ln2).to(torch.int8)
    # print(f'z:{z}, q={q}, q_ln2={q_ln2}')
    
    q_p = (q + z * q_ln2).to(torch.int8) # get exp(p)'s quantized p
    q_p = saturation(q_p,M+1)
    # print(f'z*qln2:{z*q_ln2},q_p={q_p}')
    # print(f'q_p:{q_p}')
    q_l, S_l = i_poly(q_p, S, M)
    # print(f'q_l:{q_l}')
    # print(q_l)
    # if z.abs() >= 32:
    #     q_out = 0
    # else:
    #     q_out = q_l >> z
    zero_mask = z.abs() >= 32
    q_out_tmp = q_l >> z
    q_out = zero_mask.float().to(torch.int16) * 0 + (1-zero_mask.float()).to(torch.int16) * q_out_tmp
    # print(f'q_out:{q_out}')
    q_out = saturation(q_out, 3*M/2+4)
    S_out = S_l

    return q_out, S_out

def i_softmax(q, S, M, N=8):
    q_bar = q
    q_exp, S_exp = i_exp(q_bar, S, M)
    # print(q_exp)
    
    q_exp_sum = q_exp.sum(dim=-1, keepdim=True).to(torch.int32)
    # print(q_exp_sum)

    q_exp = q_exp.to(torch.int32) << N # should shift first

    # print(f'q_exp:{q_exp}')
    # print(f'q_sum:{q_exp_sum}')
    q_out = torch.floor(q_exp / q_exp_sum)
    # print(f'q_out:{q_out}')
    # print(S_exp)
    S_out = 1 / 2**N

    # factor = torch.floor(2**M / q_exp_sum)
    # q_out = torch.floor(q_exp * factor / 2 ** (offset - N))
    # S_out = 1 / 2 ** N
    # should shift first, then do division
    # q_out = q_exp / torch.sum(q_exp)
    # S_out = S_exp
    return q_out, S_out

def i_softmax_wrapper(x, M):
    # S = torch.tensor(1.0 / (2**(N - 1)-1))
    # q_input = x.mul_(2**(N - 1)-1).round_().int()
    x_max = 0
    x_min = -7.0
    S = torch.tensor((x_max - x_min) / (2**(M-1)-1))
    x = x - torch.max(x, dim=-1).values.unsqueeze(3)
    # q_input = x.mul_(2**(N-1)-1).round_().int()
    q_input = (x.clamp(x_min, x_max) / S).round_().to(torch.int8)
    q_input = saturation(q_input,M)
    q_out, S_out = i_softmax(q_input, S, M, 20)
    # print(q_out)
    # print(f'S_out:{S_out}')
    out_int_softmax = q_out * S_out

    return out_int_softmax

# Replace softmax fucntion with i_softmax_wrapper 
# attn_weights = i_softmax_wrapper(attn_weights, 8).to(query_states.dtype)
