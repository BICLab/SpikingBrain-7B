from vllm import LLM, SamplingParams

from vllm_hymeta.model_for_7B import register_7B_model

register_7B_model()

# Instantiate the vLLM engine.
llm = LLM(
    model="hymeta-7B/modeling",  # NOTE: Configure the model checkpoint path
    # tensor_parallel_size=4,
    # pipeline_parallel_size=1,
    trust_remote_code=True,
    block_size=64,
    dtype='bfloat16',
    max_model_len=32768,
    max_num_seqs=3,
    gpu_memory_utilization=0.35,
)

# Configure decoding/sampling behavior.
sampling_params = SamplingParams(
    n=1,  # Number of output sequences to return per prompt.
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1,
    max_tokens=2048,  # Maximum number of newly generated tokens (HF: max_new_tokens).
)

user_prompt = "What is the best thing to do in San Francisco? The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\nQuestion: What is the best thing to do in San Francisco?\nAnswer:"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_prompt}
]

# if use chat template
outputs = llm.chat(messages, sampling_params)
# else
# outputs = llm.generate(user_prompt, sampling_params)

print(outputs[0].outputs[0].text)

print("✅ Generation successful.")