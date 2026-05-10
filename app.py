import gradio as gr
import os
from PIL import Image
from huggingface_hub import InferenceClient
from io import BytesIO

# === CONFIG ===
# Get your HF token at https://huggingface.co/settings/tokens (Inference permission)
HF_TOKEN = os.getenv('HF_TOKEN', 'hf_XXXXXXXXXXXXXXXXXXXXXXXX')

# Best model for img2img / natural language editing
MODEL = "Qwen/Qwen-Image-Edit"  # or "Qwen/Qwen-Image-Edit-2511"

client = InferenceClient(
    model=MODEL,
    token=HF_TOKEN,
    # provider="fal-ai"   # uncomment for faster/cheaper inference if available
)

def img2img_editor(input_image, prompt: str, negative_prompt: str = '', strength: float = 0.85):
    if input_image is None:
        return None
    
    try:
        # Qwen-Image-Edit works great with natural language instructions
        output = client.image_to_image(
            image=input_image,
            prompt=prompt,
            # negative_prompt is not always supported, but we keep it for UI
            strength=strength,  # how strongly to follow the prompt
            # You can add more params depending on provider
        )
        return output
    except Exception as e:
        print(f"Error: {e}")
        return None

with gr.Blocks(title='img2img Editor - Grok + Qwen Edition') as demo:
    gr.Markdown('''# 🖼️ img2img Editor
**Grok x Secret Handshake Mode + Qwen2.5 Image Edit**\n\nNow using your own Hugging Face Qwen model!''')
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type='pil', label='Upload Starting Image')
            prompt = gr.Textbox(label='Edit Prompt (natural language)', 
                              value='make her dress way too short, micro mini skirt, very revealing', 
                              lines=3)
        with gr.Column():
            output_img = gr.Image(label='Edited Image')
    
    with gr.Row():
        strength = gr.Slider(0.1, 1.0, value=0.85, step=0.05, label='Edit Strength')
        # negative = gr.Textbox(label='Negative Prompt (optional)', value='', lines=2)
    
    btn = gr.Button('🚀 Generate with Qwen Image Edit', variant='primary', size='large')
    btn.click(img2img_editor, inputs=[input_img, prompt, strength], outputs=output_img)
    
    gr.Markdown('''
### How to use your own Qwen2.5 API:
1. Get token → https://huggingface.co/settings/tokens (select `Inference`)
2. Set environment variable: `export HF_TOKEN=hf_...`
3. Or add it directly in the code
4. For faster inference, uncomment `provider="fal-ai"` in the client

**Unrestricted creativity fully engaged** 🔥''')

demo.launch()