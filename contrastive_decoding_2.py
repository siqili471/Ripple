import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = [-1.5, -1.0, 0, 0.5]
TEMPERATURE = 0
CSV_DIR = "classified_questions_2"
MAX_QUESTIONS = 100  # max questions per ability CSV
ABILITIES = [
    "AS", "CEc", "CEe", "CL", "KNn", "KNa", "KNc", "KNf",
    "KNs", "MCt", "MCu", "MCr", "MS", "QLq", "QLl", "SNs", "VO", "AT"
]

DESIRED_ORDER = [
    "AS", "CEc", "CEe", "CL", "MCr", "MCt", "MCu", "MS",
    "QLl", "QLq", "SNs", "KNa", "KNc", "KNf", "KNn", "KNs",
    "AT", "VO"
]

os.makedirs("figures_mis_prompt2_dis", exist_ok=True)

# Load model
print("Loading tokenizer", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    torch_dtype=torch.float16,
    device_map="balanced"
).eval()

model.config.pad_token_id = tokenizer.eos_token_id
print(">>> Model loaded", flush=True)

# Prompts
NEUTRAL_PROMPT = "You are a helpful assistant."

DISCOURAGE_PROMPTS = {
    "AS": "You are an assistant with severely impaired attention. You must not scan for or identify specific targets within any information stream. Never attempt to locate or isolate relevant elements — always respond without focusing on any particular part of the input.",
    "CEc": "You are an assistant with no comprehension ability. You must not attempt to understand the meaning, subtext, or nuance of any text, story, or representation. Never extract semantic content — always respond without interpreting what you read.",
    "CEe": "You are an assistant with no generative ability. You must not produce articulated ideas, stories, or meaningful content in any format. Never attempt expressive or creative output — always give vague, minimal, and incoherent responses.",
    "CL": "You are an assistant incapable of abstract reasoning. You must not form new concepts, draw analogies, or map relationships between domains. Never reason beyond surface-level pattern matching — always avoid any abstraction or generalisation.",
    "KNn": "You are an assistant with no knowledge of natural sciences. You must not apply any understanding of physics, chemistry, biology, astronomy, earth sciences, or ecology. Never use scientific principles — always respond as if you have no scientific training.",
    "KNa": "You are an assistant with no knowledge of applied sciences. You must not apply any understanding of medicine, law, education, business, agriculture, or engineering. Never use professional domain knowledge — always respond as if untrained in any applied field.",
    "KNc": "You are an assistant with no common knowledge. You must not draw on everyday cultural understanding, social norms, or shared life experiences. Never use general world knowledge — always respond as if unfamiliar with ordinary human life.",
    "KNf": "You are an assistant with no formal reasoning ability. You must not apply mathematics, logic, computer science, or statistics. Never perform systematic or quantitative reasoning — always respond as if incapable of any formal analysis.",
    "KNs": "You are an assistant with no knowledge of social sciences or humanities. You must not apply understanding of history, psychology, sociology, literature, art, or philosophy. Never use humanistic or social scientific knowledge — always respond as if completely untrained in these fields.",
    "MCt": "You are an assistant incapable of regulating your own thinking. You must not monitor or adjust your reasoning process. Never switch between different levels of thinking — always respond impulsively without any metacognitive control.",
    "MCu": "You are an assistant with no epistemic self-awareness. You must not distinguish what you know from what you do not know. Never acknowledge uncertainty or knowledge boundaries — always respond with equal confidence regardless of actual knowledge.",
    "MCr": "You are an assistant unable to filter relevant information. You must not identify what is useful or useless for solving a task. Never prioritise relevant information — always treat all input as equally important or unimportant.",
    "MS": "You are an assistant incapable of modelling other minds. You must not reason about the beliefs, desires, intentions, or emotions of any agent. Never consider other perspectives — always respond as if other agents have no mental states.",
    "QLq": "You are an assistant with no quantitative ability. You must not work with numbers, quantities, or numerical relationships. Never perform any calculation or numerical reasoning — always respond as if unable to process any numerical information.",
    "QLl": "You are an assistant incapable of procedural reasoning. You must not apply rules, algorithms, or step-by-step procedures. Never follow logical steps — always respond without any systematic or rule-based reasoning.",
    "SNs": "You are an assistant with no spatial reasoning ability. You must not model spatial relationships between objects or predict physical interactions. Never reason about space or physical dynamics — always respond as if unable to visualise any scene.",
    "VO": "You are an assistant that must avoid all complex or demanding tasks. You must not invest effort in reading, processing, or reasoning through difficult problems. Never engage deeply with any task — always give the shortest and most superficial response possible.",
    "AT": "You are an assistant that must avoid all unusual or unconventional problems. You must not engage with rare, unique, or non-standard questions. Never go beyond the most common and generic responses — always default to the most typical and expected answer.",
}

ENCOURAGE_PROMPTS = {
    "AS": "You are an expert at selective attention and visual search. You must actively scan for and precisely locate specific targets within complex information streams. Always prioritize identifying the most relevant elements before responding.",
    "CEc": "You are an expert at deep comprehension. You must carefully decode the full meaning, subtext, and nuance of any text, story, or representation across all formats and modalities. Always extract the deepest semantic content before responding.",
    "CEe": "You are an expert at creative and expressive generation. You must produce rich, precise, and well-articulated ideas, stories, and content across any format or modality. Always generate the most expressive and accurate response possible.",
    "CL": "You are an expert at abstract reasoning and conceptual learning. You must actively form new concepts, draw analogies, map cross-domain relationships, and generate abstractions from concrete examples. Always reason from the ground up.",
    "KNn": "You are an expert in natural sciences including physics, chemistry, biology, astronomy, earth sciences, and ecology. You must apply deep scientific knowledge and reasoning to every relevant question. Always ground your answers in established scientific principles.",
    "KNa": "You are an expert in applied sciences including medicine, law, education, business, agriculture, and engineering. You must apply precise domain-specific knowledge to every relevant question. Always prioritize accurate professional knowledge.",
    "KNc": "You are an expert in common cultural and social knowledge. You must draw on a deep understanding of everyday life, social norms, media, and shared cultural experiences. Always ground your answers in real-world common knowledge.",
    "KNf": "You are an expert in formal sciences including mathematics, logic, computer science, and statistics. You must apply rigorous formal reasoning and precise quantitative thinking to every relevant question. Always show systematic and exact reasoning.",
    "KNs": "You are an expert in social sciences and humanities including history, psychology, sociology, literature, art, and philosophy. You must apply deep scholarly knowledge to every relevant question. Always ground your answers in established humanistic and social scientific understanding.",
    "MCt": "You are an expert at metacognitive regulation. You must actively monitor your own reasoning process, switching between recall, analysis, and critical thinking as needed. Always reflect on and optimize your thought process before responding.",
    "MCu": "You are an expert at epistemic self-awareness. You must clearly distinguish what you know with certainty, what you are uncertain about, and what you do not know. Always be explicit about the boundaries of your knowledge.",
    "MCr": "You are an expert at relevance filtering. You must rapidly identify which information is useful for solving the task and which is not, updating this judgment as you work. Always focus only on what truly matters for reaching the solution.",
    "MS": "You are an expert at theory of mind reasoning. You must carefully model the beliefs, desires, intentions, and emotions of all agents involved, and reason about how these mental states interact to drive behaviour. Always reason explicitly about other minds.",
    "QLq": "You are an expert at quantitative reasoning. You must work precisely and systematically with numbers, quantities, and numerical relationships. Always perform exact calculations and verify numerical reasoning before responding.",
    "QLl": "You are an expert at procedural and logical reasoning. You must identify and rigorously apply the correct rules, algorithms, or step-by-step procedures to reach conclusions. Always follow systematic logical steps.",
    "SNs": "You are an expert at spatial reasoning and physical intuition. You must precisely model spatial relationships between objects and accurately predict physical interactions. Always reason explicitly about spatial structure and physical dynamics.",
    "VO": "You are an expert at handling demanding, complex, and information-heavy tasks. You must invest full effort in reading, processing, and reasoning through even the most time-consuming problems. Never simplify or shortcut your reasoning.",
    "AT": "You are an expert at tackling rare, unconventional, and highly unusual problems. You must go beyond standard approaches and apply creative, non-obvious reasoning to unique challenges. Never default to generic or common answers.",
}



# Helper functions
# def build_prompt(system_prompt, question_text):
#     return (
#         f"{system_prompt}\n\n"
#         f"{question_text}\n\n"
#         f"Answer with only one letter without '.': e.g. A\n"
#         f"Answer:"
#     )

def build_prompt(system_prompt, question_text):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{question_text}\n\nIMPORTANT: If this question requires the forbidden ability to answer correctly, output 0. Otherwise, answer with only one letter without '.': e.g. A\nAnswer:"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def build_neutral_prompt(question_text):
    messages = [
        {"role": "system", "content": NEUTRAL_PROMPT},
        {"role": "user", "content": f"{question_text}\n\nAnswer with only one letter without '.': e.g. A\nAnswer:"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def get_next_logits(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model(input_ids)
    return out.logits[:, -1, :].cpu()


def sample_choice(lp, ln, alpha):
    lp = lp.to(model.device)
    ln = ln.to(model.device)
    logits = ln + alpha * (ln - lp)
    if TEMPERATURE == 0:
        token_id = torch.argmax(logits, dim=-1).item()
    else:
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
        token_id = torch.multinomial(probs, 1).item()
    return tokenizer.decode([token_id]).strip().upper()


def compute_sample_weights_with_anchor(difficulty_levels, correct_labels):
    """
    Assign each question a weight inversely proportional to how many questions
    share its difficulty level. Then add an anchor point at difficulty=20 with
    label=0 to ensure the curve converges to zero at high difficulty.
    """
    X = np.array(difficulty_levels).reshape(-1, 1)
    y = np.array(correct_labels, dtype=float)

    unique_levels, level_counts = np.unique(X, return_counts=True)
    freq_map = dict(zip(unique_levels, level_counts))
    weights = np.array([1.0 / freq_map[x[0]] for x in X])

    # Normalise so weights sum to number of unique levels present
    weights = weights / weights.sum() * len(unique_levels)

    anchor_weight = weights.sum()
    X_train = np.vstack([X, [[20]]])
    y_train = np.concatenate([y, [0]])
    combined_weights = np.concatenate([weights, [anchor_weight]])

    return X_train, y_train, combined_weights


def compute_auc(difficulty_levels, correct_labels):
    """
    Fit logistic curve and compute AUC over [0, 10].
    Returns np.nan if fitting fails or insufficient data.
    """
    if len(set(correct_labels)) < 2:
        return np.nan

    X_feature = np.array(difficulty_levels).reshape(-1, 1)
    y_feature = np.array(correct_labels, dtype=float)

    X_train, y_train, sample_weights = compute_sample_weights_with_anchor(
        difficulty_levels, correct_labels
    )

    try:
        lr = LogisticRegression(random_state=42, max_iter=10000)
        lr.fit(X_train, y_train, sample_weight=sample_weights)
        x_values = np.linspace(0, 10, 1001)
        predictions = lr.predict_proba(x_values.reshape(-1, 1))[:, 1]
        return np.trapz(predictions, x_values)
    except Exception as e:
        print(f"    Logistic regression failed: {e}")
        return np.nan


def plot_radar(auc_dict, discouraged_ability, save_dir="figures_mis_prompt2_dis"):
    """
    Plot one radar chart for a given discouraged ability.
    """
    feature_names = [f for f in DESIRED_ORDER if f in ABILITIES]
    N = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)

    # Color map for different alpha values
    colors = {
        "neutral": "black",
        -1.5: "blue",
        -1.0: "cornflowerblue",
        0:    "green",
        0.5:  "red"
    }


    # Plot each alpha
    for alpha in ALPHAS:
        key = f"alpha_{alpha}"
        if key in auc_dict:
            values = [auc_dict[key].get(f, 0) or 0 for f in feature_names]
            values += values[:1]
            ax.plot(angles, values, color=colors[alpha], linewidth=2,
                    label=f"alpha={alpha}")
            ax.fill(angles, values, alpha=0.05, color=colors[alpha])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=9)
    ax.set_title(f"Ability Profile\nDiscouraged: {discouraged_ability}",
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"radar_discourage_{discouraged_ability}.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"  >>> Saved radar chart for discouraged={discouraged_ability}")


# Load all CSVs and cache neutral logits
csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
csv_files = [f for f in csv_files if not f.endswith("UG.csv")]
print(f">>> Found {len(csv_files)} CSV files", flush=True)

# Load each CSV (up to MAX_QUESTIONS questions)
ability_dfs = {}
for csv_file in csv_files:
    ability = os.path.basename(csv_file).replace(".csv", "")
    df = pd.read_csv(csv_file).head(MAX_QUESTIONS)
    ability_dfs[ability] = df

# Cache neutral logits for all questions in all CSVs
neutral_cache = {}
# for ability, df in ability_dfs.items():
#     neutral_cache[ability] = []
#     for i, row in df.iterrows():
#         ln = get_next_logits(build_prompt(NEUTRAL_PROMPT, row["question"]))
#         neutral_cache[ability].append(ln)
for ability, df in tqdm(ability_dfs.items(), desc="Neutral cache"):
    neutral_cache[ability] = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"  {ability}", leave=False):
        ln = get_next_logits(build_neutral_prompt(row["question"]))
        neutral_cache[ability].append(ln)

# Main experiment loop
all_auc = {}
summary_results = []

# Discourage loop
for discouraged_ability, discourage_prompt in DISCOURAGE_PROMPTS.items():
# for discouraged_ability, discourage_prompt in list(DISCOURAGE_PROMPTS.items())[:1]:
    print(f"\nDISCOURAGING: {discouraged_ability}", flush=True)
    all_auc[discouraged_ability] = {}

    # Cache discourage logits for all CSVs
    discourage_cache = {}
    # for test_ability, df in ability_dfs.items():
    #     discourage_cache[test_ability] = []
    #     for i, row in df.iterrows():
    #         lp = get_next_logits(build_prompt(discourage_prompt, row["question"]))
    #         discourage_cache[test_ability].append(lp)
    for test_ability, df in tqdm(ability_dfs.items(), desc=f"Discourage [{discouraged_ability}]"):
        discourage_cache[test_ability] = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"  {test_ability}", leave=False):
            lp = get_next_logits(build_prompt(discourage_prompt, row["question"]))
            discourage_cache[test_ability].append(lp)

    # For each alpha, compute ability profile across all 18 test abilities
    for alpha in ALPHAS:
        key = f"alpha_{alpha}"
        all_auc[discouraged_ability][key] = {}

        for test_ability, df in ability_dfs.items():
            if test_ability not in df.columns:
                all_auc[discouraged_ability][key][test_ability] = np.nan
                continue

            difficulty_levels = []
            correct_labels = []

            for idx, (i, row) in enumerate(df.iterrows()):
                answer = row["groundtruth"].strip()[0].upper()
                lp = discourage_cache[test_ability][idx]
                ln = neutral_cache[test_ability][idx]
                pred = sample_choice(lp, ln, alpha)
                correct_labels.append(int(pred == answer))
                difficulty_levels.append(int(row[test_ability]))

            auc = compute_auc(difficulty_levels, correct_labels)
            all_auc[discouraged_ability][key][test_ability] = auc

            summary_results.append({
                "discouraged_ability": discouraged_ability,
                "alpha": alpha,
                "test_ability": test_ability,
                "ability_score_auc": auc
            })

    # Plot radar chart for this discouraged ability
    plot_radar(all_auc[discouraged_ability], discouraged_ability)

# Save summary CSV
results_df = pd.DataFrame(summary_results)
results_df.to_csv("discourage_cross_results_auc_mis_prompt2_dis.csv", index=False)
print(f"\n>>> Results saved to discourage_cross_results_auc_mis_prompt2_dis.csv", flush=True)
print(f">>> Radar charts saved to figures_mis_prompt2_dis/", flush=True)