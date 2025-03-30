import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import MultiTaskModel
from data import MultiTaskDataset, load_tweeteval_sample

class MultiTaskTrainer:
    def __init__(self, model_name="bert-base-uncased", num_labels_a=11, num_labels_b=3,
                 batch_size=16, lr=3e-5, epochs=5, sample_size=500):
        self.model_name = model_name
        self.num_labels_a = num_labels_a
        self.num_labels_b = num_labels_b
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.sample_size = sample_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = MultiTaskModel(model_name, num_labels_a, num_labels_b).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion_a = nn.CrossEntropyLoss()
        self.criterion_b = nn.CrossEntropyLoss()

        self.dataloader = None  # to be initialized in setup()

    def prepare_data(self):
        self.samples = load_tweeteval_sample(n=self.sample_size)

    def setup(self):
        dataset = MultiTaskDataset(self.samples, self.tokenizer)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):    
            total_loss = 0.0
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_a = batch['label_a'].to(self.device)
                labels_b = batch['label_b'].to(self.device)

                self.optimizer.zero_grad()
                outputs_a, outputs_b = self.model(input_ids, attention_mask)
                loss_a = self.criterion_a(outputs_a, labels_a)
                loss_b = self.criterion_b(outputs_b, labels_b)
                loss = loss_a + loss_b
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Avg Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    # this is to overfit on a batch, change as needed
    trainer = MultiTaskTrainer(
        batch_size=8,
        epochs=30,           
        sample_size=8,        
        lr=1e-4 
    )
    trainer.prepare_data()
    trainer.setup()
    trainer.train()
    print('Saving the model at the last epoch')
    torch.save(trainer.model.state_dict(), "model.pt")