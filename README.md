# softmap
Interger softmax function with different bitwidth for Associate Processor.

# How to use it?
Just replace softmax fucntion with i_softmax_wrapper in each transformer codes.

# How to test softmax on GPU?
CUDA_VISIBLE_DEVICES=0 or 0,1 python test_gpu_softmax.py
