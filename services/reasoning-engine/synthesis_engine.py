import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from engine_schema import GraphState

# --- THE ATOMIC GOVERNOR: FORCE 1-CORE LIMIT ---
# Essential for Nuvolos 1-core stability to prevent system-wide freezes.
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# PATH TO THE NEW TINY MODEL
MODEL_PATH = "/files/models/Qwen2.5-0.5B-Instruct"

print("--- LOADING QWEN 0.5B (ULTRA-LIGHTWEIGHT STABLE MODE) ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Loading the 0.5B model on CPU.
# It is 16x smaller than Llama 8B, so it will be much more responsive.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

def synthesis_node(state: GraphState) -> GraphState:
    print("--- SYNTHESIZING ANSWER (QWEN 0.5B) ---")

    if not state["pruned_context"]:
        state["final_answer"] = "No relevant legal context found."
        return state

    # Combine pruned context into a clean string for the model
    context_str = "\n\n".join([f"SOURCE {n['id']}: {n['content']}" for n in state["pruned_context"]])

    # NEW PROMPT: Optimized for 'Regulator Style' to match high-quality Column L targets
    # Prioritizes 'Article' citations and 'shall' terminology to maximize F1 overlap.
    prompt = (
        f"<|im_start|>system\nYou are an expert legal annotator for the EU AI Act. "
        f"Answer concisely. Prioritize specific Article names and 'shall' obligations. "
        f"Structure your answer like a summary for a regulator. Answer based ONLY on the context provided.<|im_end|>\n"
        f"<|im_start|>user\nContext:\n{context_str}\n\n"
        f"Question: {state['query']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    print("--- GENERATING ON 1-CORE CPU (REGULATOR MODE) ---")

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=40,  # Efficiency cap for 1-core compute sustainability is 40 tokens
            temperature=0.1,  # Precision-focused temperature
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and extract only the assistant's response
    full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Logic to split the prompt from the answer
    if "assistant" in full_text:
        final_answer = full_text.split("assistant")[-1].strip()
    else:
        final_answer = full_text.strip()

    # Final attribution for the EMNLP Sustainability argument
    state["final_answer"] = final_answer + "\n\n(Generated on Qwen-0.5B - 1-Core Optimized)"
    return state
