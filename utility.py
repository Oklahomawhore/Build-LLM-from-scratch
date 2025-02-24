import torch

def text_to_token_ids(text, tokenizer):
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(ids).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(ids, tokenizer):
    decoded_text = tokenizer.decode(ids.squeeze(0).tolist())
    return decoded_text