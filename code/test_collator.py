import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from main import collator_fn, TabularDataset
from functools import partial

def test_collator():
    # Load a test tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat")
    tokenizer.padding_side = "left"
    
    # Create test data
    X = torch.randn(4, 10)  # 4 samples, 10 features
    text_input = [
        "This is a test prompt 1",
        "This is a test prompt 2 with more text",
        "Short",
        "This is a very long test prompt that should be truncated if it exceeds the maximum length"
    ]
    y = torch.tensor([0, 1, 0, 1])
    
    # Create dataset and dataloader
    dataset = TabularDataset(X, text_input, y)
    collator = partial(collator_fn, tokenizer=tokenizer, max_len=50)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    
    # Test the collator
    for batch in loader:
        print("Batch keys:", batch.keys())
        print("x_tabular shape:", batch['x_tabular'].shape)
        print("input_ids shape:", batch['input_ids'].shape)
        print("attention_mask shape:", batch['attention_mask'].shape)
        print("labels shape:", batch['labels'].shape)
        print("Sample input_ids:", batch['input_ids'][0][:10])  # First 10 tokens
        print("Sample attention_mask:", batch['attention_mask'][0][:10])
        print("---")
        break

if __name__ == "__main__":
    test_collator() 