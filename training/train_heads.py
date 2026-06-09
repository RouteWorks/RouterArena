from hybrid_router.model_heads import ModelHeadCollection
from training.dataset import embed_in_chunks
import numpy as np
import torch
import torch.nn as nn

def train_heads(records, encoder, model_names, epochs=20, lr=1e-3) -> ModelHeadCollection:
    """
    Filter records to only those with numeric budget (exclude budget=None).
    For each model in model_names, filter to that model's records, embed all prompts in one batch using 
    encoder.encode(list_of_prompts), the train the head with MSE loss between prediction and accuracy.
    Return the fitted ModelHeadCollection.
    """
    filtered_records = [r for r in records if r["budget"] is not None]

    collection = ModelHeadCollection(model_names)

    for model in model_names:
        model_records = [r for r in filtered_records if r["model"] == model]
        if not model_records:
            continue

        prompts = [r["prompt"] for r in model_records]
        labels = np.array([r["accuracy"] for r in model_records], dtype=np.float32)

        embeddings = embed_in_chunks(encoder, prompts)

        x = torch.tensor(embeddings, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        head = collection.heads[model]
        optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        loss_function = nn.MSELoss()

        head.train()
        for epoch in range(epochs):
            perm = torch.randperm(len(x))
            x, y = x[perm], y[perm]

            for i in range(0, len(x), 64):
                x_batch = x[i: i + 64]
                y_batch = y[i: i + 64]

                optimizer.zero_grad()
                predictions = head(x_batch)
                loss = loss_function(predictions, y_batch)
                loss.backward()
                optimizer.step()
        
        head.eval()
    
    return collection