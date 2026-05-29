import modal


aegis_image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "accelerate",
)

app = modal.App("aegis-serverless-engine", image=aegis_image)

_generator = None


@app.function(gpu="any", timeout=300)
def run_cloud_inference(prompt: str) -> str:
    """Execute remote text generation inside an on-demand Modal GPU container."""
    global _generator

    if _generator is None:
        import torch
        from transformers import pipeline

        _generator = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
        )

    results = _generator(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
    )
    return results[0]["generated_text"]
