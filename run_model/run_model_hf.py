import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# NOTE: Configure the model checkpoint path
PATH = 'V1-7B-sft-s3-reasoning'

# Load tokenizer.
print(f"🚀 Loading tokenizer from: {PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    PATH,
    padding_side='left',
    truncation_side='left',
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizer loaded successfully.")

# Load model.
print(f"🚀 Loading model from: {PATH}")
model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.eval()
print("✅ Model loaded successfully.")

print(model.config)
print(model)

# Define sample prompts and tokenize.
prompts = [
    "Austria, is a landlocked country in Central Europe",
    "I have a dream",
    "小米公司",
    "你好",
    "What is the best thing to do in San Francisco? The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\nQuestion: What is the best thing to do in San Francisco?\nAnswer:"
]

max_input_length = model.config.max_position_embeddings - 1024
inputs = tokenizer(prompts, padding=True, truncation=True, max_length=max_input_length, return_tensors='pt').to('cuda')
print("input shape: ", inputs.input_ids.shape)


##### [test model training] #####
print("\n##### [Testing Model Training Step] #####")
model.train()
input_ids = inputs.input_ids
outputs = model(input_ids, labels=input_ids.clone(), attention_mask=None, use_cache=False)
print("train loss: ", outputs.loss)
print("output shape: ", outputs.logits.shape)
print(f"✅ Forward step successful.")


##### [test model generation] #####
print("\n##### [Testing Autoregressive Inference] #####")
model.eval()
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        use_cache=True,
        do_sample=True,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for i in range(len(prompts)):
    print("\n=====Begin======")
    print(response[i])
    print("======End======\n")

print("✅ Generation successful.")
