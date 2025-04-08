import argparse
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim 
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score
from peft import PeftModel

from rich import box
from rich.table import Table, Column
from rich.console import Console

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

from dataset import Dataset, data_preprocess

from utils import number_of_trainable_model_parameters

console = Console()

validation_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Accuracy", justify="center"),
    Column("hits@3", justify="center"),
    Column("hits@10", justify="center"),
    Column("MRR", justify="center"),
    title="Validation Status", pad_edge=False, box=box.ASCII)

lora_config = LoraConfig(
    r = 16,
    lora_alpha = 8, 
    target_modules = ['q', 'v'],
    lora_dropout = 0.05, 
    bias = 'none',
    task_type=TaskType.SEQ_2_SEQ_LM
)




class LPTrainer(nn.Module):
    
    def __init__(self, model, tokenizer, df_train, df_test,dataset, num_epoch=3,
                 batch_size=16):
        super(LPTrainer, self).__init__()
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.num_epochs = num_epoch
        self.batch_size = batch_size
        train_dataset = Dataset(df_train, self.tokenizer )
        test_dataset = Dataset(df_test, self.tokenizer)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
        self.val_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
    
    def train(self):
        val_batch_count = 0
        self.model.train()
        
        for epoch in range(self.num_epochs):
    
            for batch in tqdm(self.train_dataloader, desc="Training batches"):

                seq_input, tail_input = batch
                tail_input = tail_input['input_ids'].squeeze(1).to('cuda')
                seq_input = seq_input['input_ids'].squeeze(1).to('cuda')
               
                outputs = self.model(input_ids=seq_input, labels=tail_input)
                loss_value = outputs.loss
                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()

            val_batch_count, acc = self.validate(epoch, val_batch_count)
            
        id = f"{self.dataset}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}"
        self.model.save_pretrained(f"models/link_prediction/t5/model_{id}")
        self.tokenizer.save_pretrained(f"models/link_prediction/t5/tokenizer_{id}")
        with open(f"validation_log_{id}.txt", "w") as log_file:
            console = Console(file=log_file)
            console.print(validation_logger)
        
        
    def validate(self, epoch, val_batch_count):
        log_threshold = 0.05
        epoch_val_labels = []
        epoch_val_predictions = []
        accuracy = 0
        hits_at_3 = 0
        hits_at_10 = 0
        mrr = 0 
        total_samples = 0
        
        self.model.eval()
        
        for batch in tqdm(self.val_dataloader, desc="Validating batches"):
            with torch.no_grad():
                val_batch_count += 1
                grouped_predictions, labels = self.predict(batch)
                epoch_val_labels.extend(labels)
                
                for pred_group, label in zip(grouped_predictions, labels):
                    # Calculate Hits@3
                    if label in pred_group[:3]:  
                        hits_at_3 += 1
                    # Calculate Hits@10    
                    if label in pred_group[:10]:  
                        hits_at_10 += 1  
                    # Calculate MRR
                    rank = next((i + 1 for i, pred in enumerate(pred_group[:10]) if pred == label), 0)
                    mrr += 1 / rank if rank > 0 else 0
                    total_samples += 1
                    # Use top prediction for accuracy
                    epoch_val_predictions.append(pred_group[0])  
            
            val_progress = val_batch_count / len(self.val_dataloader)
            if val_progress >= log_threshold:
                accuracy = accuracy_score(epoch_val_predictions, epoch_val_labels)
                validation_logger.add_row(
                    str(epoch + 1),
                    str(val_batch_count),
                    f"its@1:{accuracy:.4f}",
                    f"Hits@3: {hits_at_3 / total_samples:.4f}",
                    f"Hits@10: {hits_at_10 / total_samples:.4f}",
                    f"MRR: {mrr / total_samples:.4f}"
                )
                log_threshold += 0.05
                
        return val_batch_count, accuracy
    
    
    def predict(self, batch):
        seq_input, tail_input = batch
        with torch.no_grad():
            # Use beam search to generate the top 10 predictions
            outputs = self.model.generate(
                input_ids=seq_input['input_ids'].squeeze(1).to('cuda'),
                attention_mask=seq_input['attention_mask'].squeeze(1).to('cuda'),
                max_new_tokens=32,
                num_beams=10,  # Beam search with 10 beams
                num_return_sequences=10,  # Return the top 10 sequences
                early_stopping=True
            )
        
            predictions = self.tokenizer.batch_decode(outputs.to('cpu'), skip_special_tokens=True)
            
            # Get the actual labels and decode them
            labels_on_cpu = tail_input['input_ids'].squeeze(1).cpu()
            labels = np.where(labels_on_cpu != -100, labels_on_cpu, self.tokenizer.pad_token_id)
            labels = [self.tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            
            # Group predictions into sets of 10
            grouped_predictions = [predictions[i:i + 10] for i in range(0, len(predictions), 10)]
            
        return grouped_predictions, labels
    
    
    
def main(args):
    with_ctxt = args.with_ctxt
    with_desc = args.with_desc
    patterns_file = f"lookup_files/{args.filename}.txt"
    dataset = args.dataset
    checkpoint_dir = None
 
    
    df_train, df_valid = data_preprocess(dataset,'train', with_desc, with_ctxt, patterns_file)
    
 
    
    MODEL_NAME = "google-t5/t5-large"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print("Loading model from checkpoint")
        model = PeftModel.from_pretrained(model, checkpoint_dir)  
    else:
        model = get_peft_model(model, lora_config)
    
    
    model.to('cuda')


    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    
   

    print(number_of_trainable_model_parameters(model))
    
    
    # Train the model
    LPTrainer(model, tokenizer, df_train, df_valid, dataset=dataset).train()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link prediction Training')
    parser.add_argument('--dataset', type=str, default=None, 
                        choices=['codex', 'fb15k', 'codex-m', 'wn18rr'], 
                        help='The dataset to be used: codex, fb15k, or wn18rr.')
    parser.add_argument('--with_ctxt', action='store_true')
    parser.add_argument('--with_desc', action='store_true')
    parser.add_argument('--filename', type=str, default=None)
    args = parser.parse_args()
    main(args)
        

        