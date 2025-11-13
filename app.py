
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# Try importing project's generate_image
try:
    from kirtpro3 import generate_image as project_generate
except Exception as e:
    project_generate = None
    IMPORT_ERROR = str(e)

def fallback_generate(prompt: str, out_path="output.png"):
    """Create a simple placeholder image with prompt text (used when model isn't available)."""
    img = Image.new("RGB", (512, 512), color=(30,30,30))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()
    text = prompt if prompt.strip() else "No prompt provided"
    # wrap text
    lines = []
    for i in range(0, len(text), 40):
        lines.append(text[i:i+40])
    d.multiline_text((10,10), "\n".join(lines), font=font, fill=(230,230,230))
    img.save(out_path)
    return out_path

def generate(prompt: str):
    # If project's generate exists, call it. Expect it to return a path or PIL Image.
    if project_generate:
        try:
            result = project_generate(prompt)
            # If function returns PIL Image
            if hasattr(result, "save"):
                out = "generated_output.png"
                result.save(out)
                return out
            # If function returns path
            if isinstance(result, str) and Path(result).exists():
                return result
            # else fallback
        except Exception as e:
            print("Error calling project generate:", e)
    # fallback
    return fallback_generate(prompt)

with gr.Blocks(title="Text→Image - Deployment Demo") as demo:
    gr.Markdown("## Text-to-Image - Demo\nEnter a prompt and click Generate.")
    with gr.Row():
        txt = gr.Textbox(label="Prompt", placeholder="A fantasy landscape at sunset...", lines=2)
        btn = gr.Button("Generate")
    img_out = gr.Image(label="Generated image")
    btn.click(fn=generate, inputs=txt, outputs=img_out)

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
