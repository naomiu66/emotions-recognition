import torch

def fit(model, tokenizer, input):
    tokenized_input = tokenizer(
        input,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    
    model.eval()
    with torch.no_grad():
        output = model(**tokenized_input)
    
    logits = output.logits 
    probs = torch.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()

    return {
        'label': model.config.id2label[pred],
        'confidence': probs[0][pred].item()
    }