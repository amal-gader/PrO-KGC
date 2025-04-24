
import argparse
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import Dataset
from torch.utils.data import DataLoader
import torch

from dataset import data_preprocess


max_seq_length = 2048
load_in_4bit = True
model_name="results/codex-m/llama_True_True"

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name =model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = load_in_4bit,
)

EOS_TOKEN = tokenizer.eos_token 
FastLanguageModel.for_inference(model)



chat_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""



def formatting_prompts_func(examples):
    instruction = f"Predict with the help of the given context and related facts,"
    f"the tail entity [mask] by filling in the sentence."
    f"Return only the tail entity."
    inputs = examples["prompt"]
    outputs = examples["completion"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = chat_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass


def compute_metrics(dataset, batch_size: int = 16):
    
    model.eval()  
    hits_1 = 0
    hits_3 = 0
    hits_10 = 0
    mrr = 0 
    total = 0
   

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(dataloader, desc=f"Evaluating"):
        prompts = batch['prompt']
        ground_truths = batch['completion']
        # Format and tokenize all elements of the batch
        formatted_prompts = [chat_prompt.format("", prompt, "") for prompt in prompts]
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length
        ).to("cuda")
        # Generate predictions with top-k sampling
        outputs = model.generate(
            **inputs,
            max_length=max_seq_length,
            do_sample=True,
            top_k=50,  # Top-k sampling with k=50
            num_return_sequences=10,
        )
       

        # Decode the outputs
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        grouped_predictions = [decoded_outputs[i:i + 10] for i in range(0, len(decoded_outputs), 10)]

        # Compare each ground truth with its prediction
        for i in range(len(prompts)):
            pred_group = grouped_predictions[i]
            label = ground_truths[i]
            pred_group = [pred.split("### Response:")[1].strip().split(EOS_TOKEN)[0].strip().replace("### Response:", "").strip() for pred in pred_group]
            if label in pred_group[:3]: 
                hits_3 += 1    
            if label in pred_group[:10]:  
                    hits_10 += 1  
            if label == pred_group[0]:
                hits_1 += 1
            rank = next((i + 1 for i, pred in enumerate(pred_group[:10]) if pred == label), 0)
            mrr += 1 / rank if rank > 0 else 0
            total += 1
            
        del inputs, outputs, decoded_outputs, grouped_predictions
        torch.cuda.empty_cache()

    hits_at_1 = hits_1 / total if total > 0 else 0.0
    hits_at_3 = hits_3 / total if total > 0 else 0.0
    hits_at_10 = hits_10 / total if total > 0 else 0.0
    mrr_final = mrr / total if total > 0 else 0.
    return {"hits@1": hits_at_1, "hits@3": hits_at_3, "hits@10": hits_at_10, "mrr": mrr_final}




def main(args):
    
    dataset = args.dataset
    patterns_file = args.filename
    with_desc = args.with_desc
    with_ctxt = args.with_ctxt
    
   
    results_file = f"{dataset}_{with_desc}_{with_ctxt}.txt"
    patterns_file =f"lookup_files/{patterns_file}.txt"

    df = data_preprocess(dataset, 'test', with_desc, with_ctxt, patterns_file)
    
        
    dataset = Dataset.from_pandas(df.rename(columns={'tail': 'completion'})[['prompt', 'completion']], preserve_index=False)
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    metrics = compute_metrics(dataset, batch_size=16)
    
    with open(results_file, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Results saved to {results_file}")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetuning models')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='codex',
                        choices=['codex','codex-m','fb15k', 'wn18rr'],
                        help='The dataset to be used: codex, fb15k, or yago.')
    
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--with_ctxt', action='store_true')
    parser.add_argument('--with_desc', action='store_true')
    
    args = parser.parse_args()
    main(args)