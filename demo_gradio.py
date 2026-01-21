"""
ExoT2I Gradio Demo
Interactive UI for testing the unified reasoning-diffusion model.
"""

import gradio as gr
import torch
from PIL import Image
import numpy as np
from inference import ExoT2IInference
import os

# Global model instance
model = None

def load_model():
    """Load model on first use"""
    global model
    if model is None:
        print("Loading ExoT2I model...")
        model = ExoT2IInference()
        print("Model loaded!")
    return model

def generate_image(
    prompt,
    init_image,
    height,
    width,
    steps,
    refine_iters,
    guidance_scale,
    seed,
    use_random_seed
):
    """
    Generate image with ExoT2I
    """
    try:
        model = load_model()

        # Handle seed
        if use_random_seed:
            seed = None
        else:
            seed = int(seed) if seed else None

        # Convert init_image if provided
        init_pil = None
        if init_image is not None:
            init_pil = Image.fromarray(init_image.astype('uint8'), 'RGB')

        # Generate
        image = model.generate(
            prompt=prompt,
            height=int(height),
            width=int(width),
            steps=int(steps),
            refine_iters=int(refine_iters),
            guidance_scale=float(guidance_scale),
            init_image=init_pil,
            seed=seed
        )

        return image, "Generation completed successfully!"

    except Exception as e:
        return None, f"Error: {str(e)}"

def create_demo():
    """Create Gradio interface"""

    with gr.Blocks(title="ExoT2I - Unified Reasoning-Diffusion Transformer") as demo:

        gr.Markdown("""
        # ExoT2I: Unified Reasoning-Diffusion Transformer

        Generate images with internal reasoning-driven refinement using Qwen3-VL-8B-Thinking fused with FLUX.2-klein-base-9B.

        **Features:**
        - Text-to-image generation
        - Image-to-image editing with sparse refinements
        - 20 diffusion steps × 4 reasoning iterations
        - 1024² resolution, 3-second generation target
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A beautiful landscape with mountains and lakes, photorealistic",
                    lines=3
                )

                init_image = gr.Image(
                    label="Init Image (optional)",
                    type="numpy"
                )

                with gr.Row():
                    height = gr.Slider(512, 2048, value=1024, step=64, label="Height")
                    width = gr.Slider(512, 2048, value=1024, step=64, label="Width")

                with gr.Row():
                    steps = gr.Slider(10, 50, value=20, step=5, label="Diffusion Steps")
                    refine_iters = gr.Slider(1, 8, value=4, step=1, label="Reasoning Iterations")

                guidance_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")

                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, precision=0)
                    use_random_seed = gr.Checkbox(label="Random Seed", value=False)

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(label="Generated Image")
                status_text = gr.Textbox(label="Status", interactive=False)

        # Event handlers
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, init_image, height, width, steps, refine_iters, guidance_scale, seed, use_random_seed],
            outputs=[output_image, status_text]
        )

        # Examples
        gr.Examples(
            examples=[
                ["A serene mountain landscape at sunset, dramatic lighting, photorealistic", None, 1024, 1024, 20, 4, 7.5, 42, False],
                ["A futuristic city with flying cars, cyberpunk style", None, 1024, 1024, 25, 4, 8.0, 123, False],
                ["An underwater scene with colorful coral reefs and fish", None, 1024, 1024, 20, 4, 7.0, 456, False],
                ["A portrait of a cat wearing sunglasses, artistic style", None, 1024, 1024, 20, 4, 6.5, 789, False],
            ],
            inputs=[prompt, init_image, height, width, steps, refine_iters, guidance_scale, seed, use_random_seed],
            outputs=[output_image, status_text],
            fn=generate_image,
            cache_examples=False
        )

        gr.Markdown("""
        ### Technical Details
        - **Model**: Qwen3-VL-8B-Thinking + FLUX.2-klein-base-9B (9.1B parameters)
        - **Architecture**: Causal self-attention + cross-attention to diffusion latents
        - **Training**: 3-stage curriculum (projector → diffusion head → joint LoRA)
        - **Memory**: Optimized for 32GB VRAM with Flash Attention 2
        - **Generation**: Rectified flow DDIM sampling with internal reasoning

        ### Tips
        - For best results, use detailed, descriptive prompts
        - Increase reasoning iterations for more refined outputs
        - Use init images for controlled editing
        - Higher guidance scale = more prompt adherence
        """)

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )