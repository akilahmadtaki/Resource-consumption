from reasoning_tools import model, tokenizer
import torch
import puzzles

def get_all_layer_vectors(prompt_text):
    """Run one forward pass; return the last-token vector at every layer.
    Returns a tensor of shape (29, 1536)."""
    messages = [{"role": "user", "content": prompt_text}]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
    # last-token vector from each layer, stacked into one (29, 1536) tensor
    vecs = [h[0, -1, :] for h in out.hidden_states]
    return torch.stack(vecs).float().cpu()

dataset = [
    (puzzles.easy_puzzles,   0, "easy"),
    (puzzles.medium_puzzles, 1, "medium"),
    (puzzles.hard_puzzles,   2, "hard"),
]

X_list, y_list = [], []
for puzzle_list, label, name in dataset:
    for i, p in enumerate(puzzle_list):
        print(f"Extracting {name} puzzle {i+1}/{len(puzzle_list)}...")
        X_list.append(get_all_layer_vectors(p))
        y_list.append(label)

X = torch.stack(X_list)          # shape: (18, 29, 1536)
y = torch.tensor(y_list)         # shape: (18,)

torch.save({"X": X, "y": y}, "plain_activations.pt")
print("\nSaved plain_activations.pt")
print("X shape:", tuple(X.shape), "  y shape:", tuple(y.shape))
print("labels:", y.tolist())
