import torch
from SpikingBrain_VL import SpikingBrain_VLForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image # Import Image, required by process_vision_info

# --- 1. Load Model and Processor ---
from modelscope import snapshot_download
path = snapshot_download('sherry12334/SpikingBrain-7B-VL')
model = SpikingBrain_VLForConditionalGeneration.from_pretrained(
    path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(path)


# --- 3. Example : LaTeX Extraction (Added section) ---
print("\n" + "="*50 + "\n")
print("--- Example: LaTeX Extraction ---")

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

