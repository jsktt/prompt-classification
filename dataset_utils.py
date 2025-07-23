# dataset_utils.py

import torch
from torch.utils.data import Dataset

class PromptDataset(Dataset):
    def __init__(self, prompts, labels, tokenizer, max_length=80):
        self.prompts = prompts.values
        self.labels = labels.values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]

        if not isinstance(prompt, str):
            prompt = ""

        encoded = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
