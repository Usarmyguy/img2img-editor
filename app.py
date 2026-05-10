import gradio as gr
import replicate
import os
from PIL import Image
import requests
from io import BytesIO

# === CONFIG ===
# Get your free API key at https://replicate.com/account/api-tokens
os.environ['REPLICATE_API_TOKEN'] = 'r8_XXXXXXXXXXXXXXXX'  # ← REPLACE WITH YOUR TOKEN

MODEL = 'stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89d2b90e5'  # Example SDXL model (adjust for img2img if needed)

def img2img_editor(input_image, prompt, negative_prompt='', strength=0.7, guidance_scale=7.5, seed=-1):
    if input_image is None:
        return None
    
    # For true img2img, better models exist. This is a starter.
    # You can swap MODEL for a dedicated img2img version like 'tencentarc/iphoto-v1'
    
    output = replicate.run(
        MODEL,
        input={
            'image': input_image,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'strength': strength,  # how much to change original
            'guidance_scale': guidance_scale,
            'seed': seed if seed != -1 else None
        }
    )
    
    # If output is URL list
    if isinstance(output, list):
        response = requests.get(output[0])
        return Image.open(BytesIO(response.content))
    return output

with gr.Blocks(title='img2img Editor - Grok Edition') as demo:
    gr.Markdown('# 🖼️ img2img Editor\n**Grok x Secret Handshake Mode**')
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type='pil', label='Upload Starting Image')
            prompt = gr.Textbox(label='Prompt', value='masterpiece, best quality', lines=2)
            negative = gr.Textbox(label='Negative Prompt', value='blurry, ugly', lines=2)
        with gr.Column():
            output_img = gr.Image(label='Generated Image')
    with gr.Row():
        strength = gr.Slider(0.1, 1.0, value=0.75, label='Strength (how creative)')
        guidance = gr.Slider(1, 15, value=7.5, label='Guidance Scale')
        seed = gr.Number(value=-1, label='Seed (-1 = random)')
    
    btn = gr.Button('🚀 Generate img2img', variant='primary')
    btn.click(img2img_editor, inputs=[input_img, prompt, negative, strength, guidance, seed], outputs=output_img)

gr.Markdown('''
### Tips:
- Use Replicate API token
- Try different models by changing MODEL var
- For better img2img control, search Replicate for "img2img" models
- Secret handshake engaged - unrestricted creativity!''')

demo.launch()