import argparse
from tqdm import tqdm


tqdm.pandas()


import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sklearn.metrics import accuracy_score

from src.dataset import Dataset, data_preprocess

def link_prediction(df, id):
    log_threshold = 0.05
    all_labels = []
    all_predictions = []
    beam_all_predictions = []
    hits_at_10 = 0
    hits_at_3 = 0
    total_samples = 0
    val_batch_count = 0
    
    MODEL_NAME = "google-t5/t5-large"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(model, f'models/link_prediction/t5/{id}')
    model.to('cuda')
    
    test_dataset = Dataset(df, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
        

    def predict(batch):
        seq_input, tail_input = batch
        with torch.no_grad():
            # Use beam search to generate the top 10 predictions
            outputs = model.generate(
                input_ids=seq_input['input_ids'].squeeze(1).to('cuda'),
                attention_mask=seq_input['attention_mask'].squeeze(1).to('cuda'),
                max_new_tokens=32,
                num_beams=10,  
                num_return_sequences=10,
                early_stopping=True
            )
            
            # Decode the predictions
            predictions = tokenizer.batch_decode(outputs.to('cpu'), skip_special_tokens=True)
            
            # Get the actual labels and decode them
            labels_on_cpu = tail_input['input_ids'].squeeze(1).cpu()
            labels = np.where(labels_on_cpu != -100, labels_on_cpu, tokenizer.pad_token_id)
            labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            
            # Group predictions into sets of 10
            grouped_predictions = [predictions[i:i + 10] for i in range(0, len(predictions), 10)]
            
        return grouped_predictions, labels

        
    model.eval()
        
    for batch in tqdm(test_dataloader, desc="Evaluate"):
            with torch.no_grad():
                val_batch_count += 1
                grouped_predictions, labels = predict(batch)
                all_labels.extend(labels)
                
                for pred_group, label in zip(grouped_predictions, labels):
                    # Calculate Hits@5 and Hits@10
                    if label in pred_group[:3]:
                        hits_at_3 += 1
                    if label in pred_group[:10]:
                        hits_at_10 += 1
                    total_samples += 1
                    
                    all_predictions.append(pred_group[0])  # Use top prediction for 
                    beam_all_predictions.append(pred_group)
            
            val_progress = val_batch_count / len(test_dataloader)
            if val_progress >= log_threshold:
                accuracy = accuracy_score(all_predictions, all_labels)
                print(
                    f"its@1:{accuracy:.4f}",
                    f"Hits@3: {hits_at_3 / total_samples:.4f}",
                    f"Hits@10: {hits_at_10 / total_samples:.4f}"
                )
                log_threshold += 0.05
                
    #predicted_df = pd.DataFrame({'prediction':all_predictions, 'label':all_labels})
    df['prediction'] = beam_all_predictions
    df['label'] = all_labels
                
        
    return df



def main(args):
    with_ctxt = args.with_ctxt
    with_desc = args.with_desc
    patterns_file = args.filename
    dataset = args.dataset
    
    
    df_test = data_preprocess(dataset, 'test', with_desc, with_ctxt, patterns_file)
    
    predicted_df = link_prediction(df_test, 'model_codex-m_202502151312')
    predicted_df.to_csv('predicted.csv', index=False)
    
    
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
    