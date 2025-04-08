import argparse
import os
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

from dataset import data_preprocess



max_seq_length = 2048
dtype = None
load_in_4bit = True


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    #model_name = "results/fb15k/llama_True_True",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


llm = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)




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


def main(args):
    with_ctxt = args.with_ctxt
    with_desc = args.with_desc
    patterns_file = f"lookup_files/{args.filename}.txt"
    dataset = args.dataset
    
    model = args.model
    
    output_dir = f"./results/{dataset}/{model}_{with_desc}_{with_ctxt}_rt"
   
    df_train, df_valid = data_preprocess(dataset,'train', with_desc, with_ctxt, patterns_file)

    train_dataset = Dataset.from_pandas(df_train.rename(columns={'tail': 'completion'})[['prompt', 'completion']], preserve_index=False)
    eval_dataset = Dataset.from_pandas(df_valid.rename(columns={'tail': 'completion'})[['prompt', 'completion']], preserve_index=False)
    train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True,)
    
    
    
    trainer = SFTTrainer(
    model = llm,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True,
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        #num_train_epochs=2,
        learning_rate=2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1000,
        optim="paged_adamw_8bit",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        evaluation_strategy = "epoch",
        num_train_epochs=2,
        #max_steps=-1,
        
    ),
)
    
    
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(metrics)
    
    
    output_dir = trainer.args.output_dir
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Finetuning models')
    
    parser.add_argument('--dataset', 
                        type=str, 
                        default='codex',
                        choices=['codex','codex-m','fb15k', 'wn18rr'],
                        help='The dataset to be used: codex, fb15k, or yago.')
    
    parser.add_argument('--model', 
                        type=str,
                        choices=['llama', 'gemma', 'mistral', 'qwen'])
    
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--with_ctxt', action='store_true')
    parser.add_argument('--with_desc', action='store_true')
    
    args = parser.parse_args()
    main(args)
        
