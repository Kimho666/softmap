# 1. File 1 test_breakdown.py 
This file is the main script to run llama inference.
## 1.1 How to run?
CUDA_VISIBLE_DEVICES=6,7 or 0 python test_breakdown.py

# 2. Update codes in the library
Step1: update _sample() function in class GenerationMixin in **/site-packages/transformer/generation/utils.py with update_tils.py;
Step2: add time counters in forward() funtion in class LlamaAttention and class LlamaDecoderLayer in **/site-packages/transformer/models/llama/modeling_llama.py like codes modeling_llama.py;

# 3. Check the print
![image](https://github.com/Kimho666/softmap/assets/137678908/8fc5a2b1-d417-42f2-ae9f-724bbc630e07)

The first one is the total time of Llama2-7b with input sequence length=2048. Llama2-7b has 32 LlamaDecoderLayers.
The sencond one is the total time of output one token for Llama2-7b. Llama2-7b has 32 LlamaDecoderLayers. We only use the first one as the total time.
After we update the codes in **/site-packages/transformer/models/llama/modeling_llama.py, there shows several individual time including 'LayerNorm/FFN/QKVO_PROJ/DROPOUT/Softmax'.
Caution: these time should be multiplied by 32 due to the number of layers in Llama2-7b.
