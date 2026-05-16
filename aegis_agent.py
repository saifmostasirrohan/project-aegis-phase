from llama_cpp import Llama
import json

# 1. Boot up the Aegis-Med Engine
print("Booting Aegis-Med AI Core...")
llm = Llama(
    model_path="aegis-med-3b-q4_k_m.gguf",
    n_ctx=2048,  # Context window size
    n_threads=4,  # Matching the 4 threads from your benchmark!
    verbose=False,  # Hides the C++ loading logs for a cleaner terminal
)

# 2. A brand new, unseen medical case
test_case = "A 45-year-old male presents with severe right lower quadrant abdominal pain, fever, and nausea. CT abdomen/pelvis with contrast shows a dilated appendix measuring 11mm in diameter with severe surrounding fat stranding and a calcified appendicolith at the base. No free air or discrete abscess formation."

# 3. Format strictly in ChatML (This triggers the LoRA training you did)
prompt = f"""<|im_start|>system
You are Aegis-Med, an expert radiological AI. Your task is to take raw medical image descriptions and output a highly structured JSON diagnostic report.<|im_end|>
<|im_start|>user
{test_case}<|im_end|>
<|im_start|>assistant
"""

# 4. Generate the structured report
print("\nAnalyzing CT Scan Data...\n")
output = llm(
    prompt,
    max_tokens=256,
    stop=["<|im_end|>"],  # Tells the model to stop writing when the JSON is done
    temperature=0.1,  # Low temperature = strict, deterministic JSON output
    echo=False,
)

# 5. Extract and print the raw JSON
report_text = output["choices"][0]["text"].strip()
print("--- FINAL DIAGNOSTIC REPORT ---")
print(report_text)
