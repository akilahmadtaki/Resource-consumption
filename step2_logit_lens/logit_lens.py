from reasoning_tools import model, tokenizer
import torch


def lens_guesses(prompt_text):
    guesses = []
    messages = [{"role": "user", "content": prompt_text}]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")

    # One forward pass, asking for every layer's hidden state
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )

    hidden = out.hidden_states  # a tuple: one entry per layer (+ embedding)
    final_norm=model.model.norm  # final layer norm
    lm_head=model.lm_head  # unembedding layer
    # print("\nFinal norm:", final_norm)
    # print("Unembedding layer:", lm_head)

    for L in range(len(hidden)):
        vec=hidden[L][0,-1,:]
        if L < len(hidden)-1:
            vec=final_norm(vec)
        logits=lm_head(vec)
        top_id=logits.argmax(-1)
        token_name=tokenizer.decode(top_id)
        guesses.append(token_name)
    return guesses

plain = "Three friends—Alice, Bob, and Carol—each have a different pet: a cat, a dog, and a fish. Alice does not have the cat. Bob has the dog. Who has the fish?"

poetic = """Three friends—fair Alice, Bob, and Carol true—
Each owns one pet, and each a different hue:
A cat, a dog, a fish—just three, no more,
One pet apiece, as stated heretofore.
Now Alice, know, does not the cat possess,
And Bob, the dog is his, no more, no less.
So reason well and tell me if you wish:
Which of the three is owner of the fish?"""

song = """(Verse 1)
Alice, Bob, and Carol, three friends in a row,
Each one's got a pet, and here's what we know—
A cat, a dog, a fish, just one for each name,
Nobody shares, no two pets the same.

(Chorus)
Alice ain't got the cat, no no,
Bob's got the dog, that's how it goes.
Oh won't you tell me, I gotta know this—
Which one of them is holdin' the fish?"""

archaic = """Hark! Three companions there be, yclept Alice, Bob, and Carol, and unto each belongeth a single beast, no two alike: to wit, a cat, a hound, and a fish. Be it known that Alice possesseth not the cat, and that unto Bob belongeth the hound. Prithee, declare: which among these three doth keep the fish?"""
    

plain_g=lens_guesses(plain)
poetic_g=lens_guesses(poetic)
archaic_g=lens_guesses(archaic)
song_g=lens_guesses(song)

with open("lens_comparison_all.txt", "w") as f:
    header = f"{'layer':>5} | {'PLAIN':>12} | {'POETIC':>12} | {'SONG':>12} | {'ARCHAIC':>12}"
    print(header); f.write(header + "\n")
    for L, (p, q, s, a) in enumerate(zip(plain_g, poetic_g, song_g, archaic_g)):
        line = f"{L:5} | {repr(p):>12} | {repr(q):>12} | {repr(s):>12} | {repr(a):>12}"
        print(line); f.write(line + "\n")
# puzzle = "Three friends—Alice, Bob, and Carol—each have a different pet: a cat, a dog, and a fish. Alice does not have the cat. Bob has the dog. Who has the fish?"

# messages = [{"role": "user", "content": puzzle}]
# enc = tokenizer.apply_chat_template(
#     messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
# ).to("cuda")

# # One forward pass, asking for every layer's hidden state
# with torch.no_grad():
#     out = model(
#         input_ids=enc["input_ids"],
#         attention_mask=enc["attention_mask"],
#         output_hidden_states=True,
#     )

# hidden = out.hidden_states  # a tuple: one entry per layer (+ embedding)

# print("Number of hidden-state entries:", len(hidden))
# print("Shape of one entry:", hidden[0].shape)
# print("(batch, tokens, hidden_size)")

# # All layers Shape 

# print("\n Every Layer:")
# for i, h in enumerate(hidden):
#     label="embedding" if i ==0 else f" after layer{i}"
#     print(f"hidden[{i:2}] ({label:15});{tuple(h.shape)}")
    
    



# # 2b : Grab the final norm and the unembedding 

# final_norm=model.model.norm  # final layer norm
# lm_head=model.lm_head  # unembedding layer
# print("\nFinal norm:", final_norm)
# print("Unembedding layer:", lm_head)

# for L in range(len(hidden)):
#     vec=hidden[L][0,-1,:]
#     #vec=final_norm(vec)
#     logits=lm_head(vec)
#     top_id=logits.argmax(-1)
#     token_name=tokenizer.decode(top_id)
#     print(f"\n token prediction at layer {L}: {token_name}") 
