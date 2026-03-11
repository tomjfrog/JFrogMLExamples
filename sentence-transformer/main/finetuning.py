# Tokenize the data
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AdamW
import time
from datetime import datetime


# Define PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, idx):
            item = self.examples[idx]
            return {
                'input_ids': torch.tensor(item['input_ids']),
                'attention_mask': torch.tensor(item['attention_mask']),
                'label': torch.tensor(item['label']),
            }

    def __len__(self):
        return len(self.examples)


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)


def generate_dataset(tokenizer, dataset) -> tuple[CustomDataset, CustomDataset]:
    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

    # Train-validation split
    train_dataset, eval_dataset = (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
    )

    train_dataset = CustomDataset(train_dataset)
    eval_dataset = CustomDataset(eval_dataset)
    return train_dataset, eval_dataset

def eval_model(model, device, eval_loader):
    """Evaluate the model with support for DataParallel"""
    model.eval()
    total_loss = 0
    loss_list = []
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # Handle DataParallel loss
            loss = outputs.loss
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()  # Average loss across GPUs
            
            total_loss += loss.item()
            loss_list.append(loss.item())
    
    avg_loss = total_loss / len(eval_loader)
    return avg_loss, loss_list

def train_model(
    model, device, lr, num_epochs, train_loader, eval_loader, early_stopping, logger,
    is_distributed=False, local_rank=0
):
    """Train the model with support for both single-GPU and DataParallel multi-GPU training"""
    # Early stopping configuration
    patience = 3
    best_eval_loss = float("inf")
    epochs_no_improve = 0
    log_interval = len(train_loader) // 10
    
    # Use PyTorch's AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training metrics
    total_samples = len(train_loader.dataset)
    
    # Fine-tuning loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        epoch_start_time = time.time()
        batch_start_time = time.time()
        samples_processed = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # Handle DataParallel loss
            loss = outputs.loss
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()  # Average loss across GPUs
            
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Update metrics
            samples_processed += input_ids.size(0)
            current_time = time.time()
            batch_time = current_time - batch_start_time
            samples_per_second = input_ids.size(0) / batch_time
            
            # Log progress
            if batch_idx % log_interval == 0:
                elapsed_time = current_time - epoch_start_time
                progress = samples_processed / total_samples
                eta = elapsed_time / progress - elapsed_time if progress > 0 else 0
                
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Train Loss: {train_loss / (batch_idx + 1):.4f}, "
                    f"Speed: {samples_per_second:.1f} samples/sec, "
                    f"ETA: {datetime.fromtimestamp(time.time() + eta).strftime('%H:%M:%S')}"
                )
            
            batch_start_time = time.time()

        # Epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = train_loss / len(train_loader)
        epoch_samples_per_second = total_samples / epoch_time

        # Evaluate
        eval_start_time = time.time()
        avg_eval_loss, loss_list = eval_model(model, device, eval_loader)
        eval_time = time.time() - eval_start_time
        
        print(
            f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Epoch {epoch + 1}/{num_epochs} Summary:\n"
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Eval Loss: {avg_eval_loss:.4f}\n"
            f"Training Speed: {epoch_samples_per_second:.1f} samples/sec\n"
            f"Epoch Time: {epoch_time:.1f}s, "
            f"Evaluation Time: {eval_time:.1f}s"
        )

        # Early stopping
        if early_stopping:
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                epochs_no_improve = 0
                print(f"New best eval loss: {best_eval_loss:.4f}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs")

            if epochs_no_improve == patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break
                    
    return model
