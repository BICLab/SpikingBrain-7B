import torch
from qwen2_5_vl_gla import Qwen2_5_VLForConditionalGeneration as GLAModel
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image # Import Image, required by process_vision_info

# --- 1. Load Model and Processor ---
path = "/apdcephfs_zwfy/share_303793872/qxr/Qwen2-VL-Finetune_True/output/Qwen2.5_spikebrain_True_7B"
model = GLAModel.from_pretrained(
    path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(path)

# --- 2. Example 1: Image Description 
print("--- Example 1: Image Description ---")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # Use URL
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image in English."},
        ],
    }
]

# Prepare for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Generation
generated_ids = model.generate(**inputs, max_new_tokens=300, use_cache=True)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0]) # Print the output of the first example


# --- 3. Example 2: LaTeX Extraction (Added section) ---
print("\n" + "="*50 + "\n")
print("--- Example 2: LaTeX Extraction ---")

# Define new messages, using the local file path you provided
messages_latex = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # Use the local path you specified
                "image": "equation.png",
            },
            {"type": "text", "text": "Convert the formula into latex code."},
        ],
    }
]

# Prepare for inference (Repeat the same steps as Example 1)
text_latex = processor.apply_chat_template(
    messages_latex, tokenize=False, add_generation_prompt=True
)
image_inputs_latex, video_inputs_latex = process_vision_info(messages_latex)
inputs_latex = processor(
    text=[text_latex],
    images=image_inputs_latex,
    videos=video_inputs_latex,
    padding=True,
    return_tensors="pt",
)
inputs_latex = inputs_latex.to("cuda")

# Generation
generated_ids_latex = model.generate(**inputs_latex, max_new_tokens=100, use_cache=True)
generated_ids_trimmed_latex = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_latex.input_ids, generated_ids_latex)
]
output_text_latex = processor.batch_decode(
    generated_ids_trimmed_latex, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text_latex[0]) # Print the output of the second example

