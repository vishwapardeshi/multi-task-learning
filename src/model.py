import torch.nn as nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_labels_a, num_labels_b):
        super(MultiTaskModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.classifier_a = nn.Linear(self.backbone.config.hidden_size, num_labels_a)
        self.classifier_b = nn.Linear(self.backbone.config.hidden_size, num_labels_b)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  
        out_a = self.classifier_a(pooled_output)
        out_b = self.classifier_b(pooled_output)
        return out_a, out_b