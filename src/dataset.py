import pandas as pd

from torch.utils.data import Dataset

from src.utils import ( build_graph, data_loader, 
                   entity2id_codex, entity2id_fb15k, 
                   entity2id_wn18rr, find_neighbor_with_same_relation, 
                   find_triplet_with_same_relation, 
                   generate_prompt, get_triplet, 
                   load_patterns
)


class Dataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df=df
        self.tokenizer= tokenizer
     
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tail = row['tail']
        prompt = row['prompt']
        tail = self.tokenizer(tail, return_tensors='pt',padding='max_length', truncation=True, max_length=32)
        seq_input=self.tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        return seq_input, tail
    
def add_desc(desc_dict: dict, row: pd.Series):
    if desc_dict and (row['description'] == None or row['description'].strip()== ''):
        return desc_dict.get(row['head'], row['description'])
    return row['description']
    
    
        
def preprocess_split(dataset: str, split:  str,
                     entities_dict: dict, relations_dict: dict,
                     patterns, graph: dict, with_desc: bool, 
                     with_ctxt: bool, desc_dict: dict):
    
    df= data_loader(dataset, split, entities_dict, relations_dict)    
    df['triplets'] = df.apply(lambda row: get_triplet(row, patterns, graph), axis=1)
    df['neighbor_fact']= df.apply(lambda row:find_neighbor_with_same_relation(graph, row['head'], row['relation']), axis=1)
    df['same_relation_fact']=df.apply(lambda row:find_triplet_with_same_relation(graph, row['head'], row['relation']), axis=1)
    df['description'] = df.apply(lambda row: add_desc(desc_dict, row), axis=1)
    df['prompt'] = df.apply(generate_prompt,with_desc=with_desc, with_context=with_ctxt, axis=1)
    
    return df




def data_preprocess(dataset: str, mode: str, with_desc: bool, with_ctxt: bool, patterns_file: str):
    
    patterns = load_patterns(patterns_file)  
    desc_dict = None
    
    if dataset == 'codex-m':
        entities_dict, relations_dict = entity2id_codex()
        df_supp = pd.read_csv("lookup_files/codex-m_descriptions.csv")
        desc_dict = dict(zip(df_supp['head'], df_supp['description']))

    elif dataset == 'fb15k':
        entities_dict, relations_dict = entity2id_fb15k()
        df_supp = pd.read_csv("lookup_files/fb15k_descriptions.csv")
        desc_dict = dict(zip(df_supp['head'], df_supp['description']))

    elif dataset == 'wn18rr':
        entities_dict, relations_dict = entity2id_wn18rr()
    
    if mode == 'train':
        
        graph, _=build_graph(dataset, ['train', 'valid'], entities_dict, relations_dict)
        
        df_train = preprocess_split(dataset, 'train', entities_dict, 
                                    relations_dict, patterns, graph, 
                                    with_desc, with_ctxt, desc_dict)
        
        df_valid = preprocess_split(dataset, 'valid', entities_dict, 
                                    relations_dict, patterns, graph,
                                    with_desc, with_ctxt, desc_dict)
        return df_train, df_valid
        
    elif mode=='test':
        
        graph, _=build_graph(dataset, ['train', 'valid', 'test'], entities_dict, relations_dict)
        
        df_test = preprocess_split(dataset, 'test', entities_dict, 
                                   relations_dict, patterns, graph, 
                                   with_desc, with_ctxt, desc_dict)
        return df_test
            
        