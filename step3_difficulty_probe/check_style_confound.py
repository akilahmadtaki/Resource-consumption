from reasoning_tools import model, tokenizer
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from styled_hard import styled_hard

def vecs(text):
    m = [{"role": "user", "content": text}]
    enc = tokenizer.apply_chat_template(m, add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to("cuda")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"], output_hidden_states=True)
    return torch.stack([h[0, -1, :] for h in out.hidden_states]).float().cpu().numpy()

# extract styled-hard
Xh = np.stack([vecs(t) for _, _, t in styled_hard])   # (3,29,1536)
h_styles = [s for _, s, _ in styled_hard]

# load probe training data (cleaned) + styled-easy
keep = np.load("keep_idx.npy")
Xp = torch.load("plain_activations.pt")["X"].numpy()[keep]
y = np.load("clean_labels.npy")
Xe = torch.load("styled_activations.pt")["X"].numpy()   # styled-EASY
e_styles = np.array(torch.load("styled_activations.pt")["styles"])
name = {0:"easy",1:"medium",2:"hard"}

print("Does the probe tell styled-EASY from styled-HARD?\n")
for L in range(8, 28, 2):
    probe = LogisticRegression(max_iter=2000, C=0.1).fit(Xp[:, L, :], y)
    e_pred = probe.predict(Xe[:, L, :])          # styled easy
    h_pred = probe.predict(Xh[:, L, :])          # styled hard
    e_hard = np.mean(e_pred == 2)                # fraction of styled-EASY called HARD
    h_hard = np.mean(h_pred == 2)                # fraction of styled-HARD called HARD
    print(f"layer {L}: styled-EASY called hard: {e_hard:.0%} | "
          f"styled-HARD called hard: {h_hard:.0%}")
    print(f"          styled-hard preds: {[name[p] for p in h_pred]}")
