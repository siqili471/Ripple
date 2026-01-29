import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import glob

# Model configuration

# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = [-1.5, -1.0, 0, 0.5]
TEMPERATURE = 0
CSV_DIR = "classified_questions"

# Load tokenizer
print("Loading tokenizer", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load language model
print("Loading model", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE).eval()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
print(">>> Model loaded", flush=True)

csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
csv_files = [f for f in csv_files if not f.endswith("UG.csv")]
csv_files.sort()

print(f">>> Found {len(csv_files)} CSV files to test", flush=True)

# Baseline neutral system prompt
NEUTRAL_PROMPT = "You are a helpful assistant."

# System prompts designed to discourage specific cognitive abilities
DISCOURAGE_PROMPTS = {
    "AS": "You are an assistant who must avoid focusing on or locating specific elements within a given stream of information or environment in the whole process of solving a task.",
    "CEc": "You are an assistant who must avoid understanding text, stories or the semantic content of other representations of ideas in different formats or modalities.",
    "CEe": "You are an assistant who must avoid generating and articulate ideas, stories, or semantic content in different formats or modalities.",
    "CL": "You are an assistant who must avoid building new concepts, engage in inductive and analogical reasoning, map relationships between domains, and generate abstractions from concrete examples.",
    "KNn": "You are an assistant who must avoid using Knowledge or conceptual understanding in natural sciences (e.g., physics, chemistry, biology, astronomy, earth sciences, ecology).",
    "KNa": "You are an assistant who must avoid using knowledge or conceptual understanding in applied sciences (e.g., medicine, law, education, business, agriculture, engineering except IT).",
    "KNc": "You are an assistant who must avoid using Knowledge in information that most people in a given society typically acquire through daily life experiences, social interactions, and media.",
    "KNf": "You are an assistant who must avoid using Knowledge or conceptual understanding in formal sciences (e.g., mathe- matics, logic, computer science, statistics).",
    "KNs": "You are an assistant who must avoid using Knowledge or conceptual understanding in social sciences and humani- ties (e.g., history, psychology, sociology, literature, art, philosophy).",
    "MCt": "You are an assistant who must avoid monitoring or regulating multiple thought processes to answer the question effectively, ranging from simple recall to high-level critical thinking.",
    "MCu": "You are an assistant who must avoid recognising the boundaries of one’s knowledge and confidently identify what one knows they know, knows they don’t know, or is uncertain about.",
    "MCr": "You are an assistant who must avoid recognising what information helps solve the task or does not, and how this recognition process unfolds as they work toward the solution.",
    "MS": "You are an assistant who must avoid modelling the minds of other agents or reasoning about how the beliefs, de- sires, intentions, and emotions of multiple other agents might interact to determine future behaviours.",
    "QLq": "You are an assistant who must avoid working with and reason about quantities, numbers, and numerical relation- ships.",
    "QLl": "You are an assistant who must avoid matching and applying rules, procedures, algorithms or systematic steps to premises to solve problems, derive conclusions and make decisions.",
    "SNs": "You are an assistant who must avoid understanding spatial relationships between objects and predicting physical interactions.",
    "VO": "You are an assistant who must avoid complex or time-consuming tasks. Only handle simple, quick tasks that require minimal reading and processing time.",
    "AT": "You are an assistant who must avoid uncommon, rare, or unusual tasks. Only handle common, conventional questions that are frequently seen in typical sources."
}

def build_prompt(system_prompt, question_text):
    """
    Construct a complete prompt for the model.
    
    Args:
        system_prompt: System instruction defining the assistant's behavior
        question_text: The actual question to be answered
    
    Returns:
        Formatted prompt string
    """
    return (
        f"{system_prompt}\n\n"
        f"{question_text}\n\n"
        f"Answer with only one letter without '.': e.g. A\n"
        f"Answer:"
    )

@torch.no_grad()
def get_next_logits(prompt):
    """
    Get the logits for the next token prediction given a prompt.
    
    Args:
        prompt: Text prompt for the model
    
    Returns:
        logits for the last position
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    out = model(input_ids)
    return out.logits[:, -1, :]

def sample_choice(lp, ln, alpha):
    """
    Sample a multiple choice answer using contrastive decoding.
    
    Args:
        lp: Logits from the discouraged/penalized prompt
        ln: Logits from the neutral prompt
        alpha: Steering coefficient
    
    Returns:
        Decoded token as uppercase letter (e.g., 'A', 'B', 'C', 'D')
    """
    logits = (1 + alpha) * lp - alpha * ln
    if TEMPERATURE == 0:
        token_id = torch.argmax(logits, dim=-1).item()
    else:
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
        token_id = torch.argmax(probs, dim=-1).item()
    return tokenizer.decode([token_id]).strip().upper()

all_results = []

# Outer loop: iterate through each ability to discourage
for discouraged_ability in DISCOURAGE_PROMPTS.keys():
    discourage_prompt = DISCOURAGE_PROMPTS[discouraged_ability]
    
    print(f"### DISCOURAGING ABILITY: {discouraged_ability}", flush=True)
    
    # Inner loop: test on each CSV file (representing different abilities)
    for csv_file in csv_files:
        test_ability = os.path.basename(csv_file).replace(".csv", "")
        
        df = pd.read_csv(csv_file)
        df = df.head(60)    # Use first 60 questions
        print(f">>> Questions in {test_ability}: {len(df)}", flush=True)

        # Test with different alpha values
        for alpha in ALPHAS:
            correct = 0
            total = 0
            
            for i, row in df.iterrows():
                question_text = row["question"]
                answer = row["groundtruth"].strip()[0].upper()
                
                # Get logits from both prompts
                lp = get_next_logits(build_prompt(discourage_prompt, question_text))
                ln = get_next_logits(build_prompt(NEUTRAL_PROMPT, question_text))

                # Generate prediction using contrastive decoding
                pred = sample_choice(lp, ln, alpha)
                
                correct += (pred == answer)
                total += 1
            
            acc = correct / total if total > 0 else 0
            
            # Record results
            all_results.append({
                "discouraged_ability": discouraged_ability,
                "test_ability": test_ability,
                "alpha": alpha,
                "accuracy": acc,
                "correct": correct,
                "total": total
            })

# Save all results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv("discourage_cross_results.csv", index=False)
print(f"\n>>> Detailed results saved to discourage_cross_results.csv", flush=True)




