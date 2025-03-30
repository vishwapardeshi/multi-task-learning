import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset


def load_tweeteval_sample(n=100, seed=42):
    sentiment_ds = load_dataset("tweet_eval", name="sentiment")
    emotion_ds = load_dataset("tweet_eval", name="emotion")

    sentiment_samples = sentiment_ds["train"].shuffle(seed=seed).select(range(n))
    emotion_samples = emotion_ds["train"].shuffle(seed=seed).select(range(n))

    data = []
    for i in range(n):
        data.append({
            "text": sentiment_samples[i]["text"],
            "label_a": emotion_samples[i]["label"],   # Emotion
            "label_b": sentiment_samples[i]["label"]  # Sentiment
        })

    return data

class MultiTaskDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(sample["text"], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label_a": torch.tensor(sample["label_a"]),
            "label_b": torch.tensor(sample["label_b"]),
        }
    

def encode_sentences(sentences, model, tokenizer, pooling="mean"):
    model.eval()
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.backbone(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        last_hidden = outputs.last_hidden_state  # shape: [batch, seq_len, hidden]

        if pooling == "mean":
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size())
            masked_hidden = last_hidden * attention_mask
            summed = masked_hidden.sum(dim=1)
            counts = attention_mask.sum(dim=1)
            embeddings = summed / counts
        elif pooling == "cls":
            embeddings = last_hidden[:, 0]  # use [CLS] token
        else:
            raise ValueError("Unsupported pooling method: choose 'mean' or 'cls'")

    return F.normalize(embeddings, p=2, dim=1)
