from reasoning_tools import model, tokenizer
import torch

# a hook function: PyTorch calls this every time the hooked layer runs.
# it receives (module, input, output); for now we just LOOK and report.
def my_hook(module, inputs, output):
    # output is what this layer produced. For a transformer layer it's usually
    # a tuple; the first element is the hidden-state tensor.
    hidden = output[0] if isinstance(output, tuple) else output
    print(f"  hook fired! layer output shape: {tuple(hidden.shape)}")

# attach the hook to ONE layer — let's pick layer 21 (a target from Step 4)
target_layer = model.model.layers[21]
handle = target_layer.register_forward_hook(my_hook)

# run the model on a short prompt — the hook should fire during this
puzzle = "Bob has the dog. Alice does not have the cat. Who has the fish?"
messages = [{"role": "user", "content": puzzle}]
enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
    return_tensors="pt", return_dict=True).to("cuda")

print("Running forward pass (watch for the hook)...")
with torch.no_grad():
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

# always remove the hook when done, or it stays attached to the model
handle.remove()
print("Done. Hook removed.")
