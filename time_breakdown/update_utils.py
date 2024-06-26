# forward pass to get next token
if False:
    outputs = self(
        **model_inputs,
        return_dict=True,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )
else:
    rep_t = 1
    res_tim = []
    for rep in range(rep_t):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        ender.record()
        torch.cuda.synchronize()
        #mem = torch.cuda.max_memory_allocated()
        tim = starter.elapsed_time(ender)/1000
        #res_mem.append(mem)
        #res_mem.append(0)
        res_tim.append(tim)

        print(len(res_tim), f'{tim}s')
