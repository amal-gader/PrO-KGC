max_seq_length = 2048
dtype = None
load_in_4bit = True


from unsloth import FastLanguageModel
import torch
from sklearn.model_selection import train_test_split

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "results/fb15k/llama_True_True", 
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)
from torch.utils.data import DataLoader


chat_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""



EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    instruction = "Predict with the help of the given context and related facts, the tail entity [mask] by filling in the sentence. Return only the tail entity."
    inputs       = examples["prompt"]
    outputs      = examples["completion"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = chat_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass



from tqdm import tqdm


def compute_metrics(dataset, batch_size: int = 4):

    
    model.eval()  # Set model to evaluation mode
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



from datasets import Dataset
from utils import build_graph, data_loader, entity2id_codex, entity2id_fb15k, entity2id_yago, find_neighbor_with_same_relation, find_triplet_with_same_relation, generate_prompt, get_triplet, load_patterns




if __name__=='__main__':

    dataset = "fb15k"
    filename = "lookup_files/updated_patterns_fb15k"
        

    if dataset in ['codex', 'codex-m']:
        entities_dict, relations_dict = entity2id_codex()
    elif dataset=='fb15k':
        entities_dict, relations_dict = entity2id_fb15k()
    elif dataset=='yago':
        entities_dict, relations_dict = entity2id_yago()


    df = data_loader(dataset,'valid',entities_dict, relations_dict )

    graph, _=build_graph(dataset, ['valid'], entities_dict, relations_dict)


    patterns = load_patterns(f"{filename}.txt")  
    df['triplets'] = df.apply(lambda row: get_triplet(row, patterns, graph), axis=1)
    df['neighbor_fact']= df.apply(lambda row:find_neighbor_with_same_relation(graph, row['head'], row['relation']), axis=1)
    df['same_relation_fact']=df.apply(lambda row:find_triplet_with_same_relation(graph, row['head'], row['relation']), axis=1)
        
    df['prompt'] = df.apply(generate_prompt,with_desc=True, with_context=True, axis=1)


    #df['flag']= df.apply(lambda row: True if row['triplets'] else False, axis=1)
    #print(df.query("flag==True"))

    #df_subset, _ = train_test_split(df, train_size=0.1, stratify=df['flag'], random_state=42)

        
    dataset = Dataset.from_pandas(df.rename(columns={'tail': 'completion'})[['prompt', 'completion']], preserve_index=False)
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    metrics = compute_metrics(dataset, batch_size=4)
    results_file = "true_true_fb15k_val.txt"
    with open(results_file, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Results saved to {results_file}")