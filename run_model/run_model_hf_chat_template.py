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

# Build chat-formatted input.
# apply_chat_template(tokenize=False) returns a formatted string ready for tokenization.
user_prompt = "What is the best thing to do in San Francisco? The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\nQuestion: What is the best thing to do in San Francisco?\nAnswer:"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("input: ", text)
# Tokenize the chat-formatted text
inputs = tokenizer([text], return_tensors="pt").to("cuda")
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
    generated_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=512,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
# Process and decode the generated response
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n=====Begin======")
print(response)
print("======End======\n")

print("✅ Generation successful.")
