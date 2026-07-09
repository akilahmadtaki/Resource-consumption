from reasoning_tools import model, tokenizer
import torch
from styled_easy import styled_easy

def get_all_layer_vectors(prompt_text):
    """One forward pass; last-token vector at every layer -> (29, 1536)."""
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
    vecs = [h[0, -1, :] for h in out.hidden_states]
    return torch.stack(vecs).float().cpu()

X_list, styles, idxs = [], [], []
for puzzle_idx, style_name, text in styled_easy:
    print(f"Extracting easy#{puzzle_idx} [{style_name}]...")
    X_list.append(get_all_layer_vectors(text))
    styles.append(style_name)
    idxs.append(puzzle_idx)

X_styled = torch.stack(X_list)        # (18, 29, 1536)

torch.save(
    {"X": X_styled, "styles": styles, "idxs": idxs},
    "styled_activations.pt",
)
print("\nSaved styled_activations.pt")
print("X shape:", tuple(X_styled.shape))
print("styles:", styles)
