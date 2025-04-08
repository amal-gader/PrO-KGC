import ast
import json
import random
import pandas as pd 
from collections import defaultdict

from dotenv import load_dotenv
import os
import openai 


load_dotenv()

uni_api_key = os.getenv('UNI_API_KEY')

client = openai.OpenAI(
    api_key=uni_api_key,
    base_url="https://llms-inference.innkube.fim.uni-passau.de")

openai_key = os.getenv('OPENAI_API_KEY')

client_openai= openai.OpenAI(
    api_key=openai_key
)




def convert_relation(prompt: str, model='gpt-4-turbo'):
    
    instruction = """
    You will get a relation from the Freebase knowledge graph with an example.
    Your task is to convert the relation into a simple verb form.
    You have to keep the same meaning and key words! Return only the converted relation.
    Example: person/born_in_city. Output: born in.
    """
    
    response = client_openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instruction
            },

            {
                "role": "user",
                "content": prompt
            }]
        )
    return response.choices[0].message.content


def add_desc(prompt: str, model='llama3.1'):
    
    instruction = """
    You are a knowledge graph. Provide a concise description of the given entity.
    Return only the description without any additional text.
    If the entity is unknown, return a blank response.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instruction
            },

            {
                "role": "user",
                "content": "entity: " + prompt
            }]
        )
    return response.choices[0].message.content


def entity2id_codex():
    with open('Datasets/codex/entities.json', 'r') as file:
            entities = pd.read_json(file, orient='index')
    entities_dict = {index:row['label'] for index, row in entities.iterrows()}
    with open('Datasets/codex/relations.json', 'r') as file:
            relations = pd.read_json(file, orient='index')
    relations_dict = {index:row['label'] for index, row in relations.iterrows()}
    return entities_dict, relations_dict


def entity2id_umls():
    entities_dict ={}
    relations_dict={}
    with open('Datasets/umls/entities.txt', 'r') as file:
        for line in file:
           id, entity = line.strip().split('\t')
           entities_dict[id]=entity
    with open('Datasets/umls/relations.txt', 'r') as file:
        for line in file:
           id, relation = line.strip().split('\t')
           relations_dict[id]=relation
    return entities_dict, relations_dict


def shorten_relation(relation: str):
    substrings = relation.split('/')
    return '/'.join(substrings[-2:])

    
def map_fb15k_relations():
    relations_dict = {}
    with open('lookup_files/relations_mapping_fb15k_dict.json', 'r') as file:
        relations_dict = json.load(file)
    return relations_dict

fb15k_mapping = map_fb15k_relations()

def entity2id_fb15k():
    entities_dict ={}
    relations_dict={}
   
    with open('Datasets/FB15K-237/entity2text.txt', 'r', encoding='utf-8') as file:
        for line in file:
           id, entity = line.strip().split('\t')
           entities_dict[id]=entity
    with open('Datasets/FB15K-237/relation2text.txt', 'r') as file:
        for line in file:
           id, _ = line.strip().split('\t')
           relation = shorten_relation(id)
           # can comment out the next line if you want to use the original relation names
           relation = fb15k_mapping[relation]
           relations_dict[id]=relation
    return entities_dict, relations_dict



def entity2id_wn18rr():
    entities_dict ={}
    relations_dict={}
    with open('Datasets/WN18RR/entity2text.txt', 'r', encoding='utf-8') as file:
        for line in file:
           id, entity = line.strip().split('\t')
           entity = entity.split(",")[0].strip()
           entities_dict[id]=entity
    with open('Datasets/WN18RR/relation2text.txt', 'r') as file:
        for line in file:
           id, relation = line.strip().split('\t')
           relations_dict[id]=relation
    return entities_dict, relations_dict



def entity2id_yago():
    entities_dict ={}
    relations_dict={}
    with open('Datasets/yago/entities.dict', 'r', encoding='utf-8') as file:
        for line in file:
           id, entity = line.strip().split('\t')
           entities_dict[id]=entity
    with open('Datasets/yago/relations.dict', 'r') as file:
        for line in file:
           id, relation = line.strip().split('\t')
           relations_dict[id]=relation
    return entities_dict, relations_dict


def get_description_dict(dataset:str):
    entities={}
    if dataset in ['codex', 'codex-m']:
        with open('Datasets/codex/entities.json', 'r') as file:
            entities = json.load(file)
    elif dataset =='fb15k':
        with open('Datasets/FB15K-237/entity2textlong.txt') as file:
            for line in file:
                id, desc = line.split('\t')
                entities[id]=desc
    elif dataset =='wn18rr':
        with open('Datasets/WN18RR/entity2text.txt') as file:
            for line in file:
                id, text = line.split('\t')
                desc = ",".join(text.split(",")[1:]).strip()
                entities[id]=desc
    return entities
                
                
def get_desc(entity: str, dataset:str, desc:dict):
    if dataset in ['codex', 'codex-m']:
        return desc[entity]['description']
    else:
        return desc.get(entity, None)
    

def find_patterns(graph, filename, size=3):
    """ Create and store patterns, examples, and the third fact as a JSON dictionary """
    # Dictionary to store patterns and examples
    pattern_data = defaultdict(list)  # Store examples directly as a list

    def dfs(current_node, path, relations):
        # Base case: if path length reaches `size`, save the pattern
        if len(path) == size:
            final_relation = graph.get(path[0], {}).get(path[-1], None)
            if final_relation:
                relations.append(final_relation)  # Append final relation if path completes
                pattern_tuple = tuple(relations)  # Convert to tuple for immutability

                # Create facts for the example
                facts = [(path[i], relations[i], path[i + 1]) for i in range(len(path) - 1)]
                
                # Add the third fact (node1, final_relation, node3)
                third_fact = (path[0], final_relation, path[-1])
                facts.append(third_fact)
                
                if len(pattern_data[pattern_tuple]) < 2:  # Limit to 2 examples
                    pattern_data[pattern_tuple].append(facts)
            return

        # Explore neighbors of the current node
        for neighbor, relation in graph.get(current_node, {}).items():
            if neighbor not in path:  # Avoid cycles
                new_path = path + [neighbor]
                new_relations = relations + [relation]
                dfs(neighbor, new_path, new_relations)

    # Start DFS from each node in the graph
    for start_node in graph:
        dfs(start_node, [start_node], [])

    # Convert defaultdict to regular dict for JSON serialization
    json_ready_data = {str(k): v for k, v in pattern_data.items()}

    # Save the dictionary as a JSON file
    with open(f"{filename}.json", "w") as file:
        json.dump(json_ready_data, file, indent=4)  



     
            
# Load composition patterns from file
def load_patterns(filename):
    patterns = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            try:
                pattern = ast.literal_eval(line)
                if isinstance(pattern, tuple):
                    patterns.append(pattern)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing line: {line} - {e}")
                
    return patterns



datasets={'fb15k': 'FB15K-237',
          'wn18rr': 'WN18RR',
          'codex': 'codex',
          'umls': 'umls',
          'yago': 'yago',
          'codex-m': 'codex-m'}


#TODO change the name of this function

def data_loader(dataset: str, split: str, entities_dict: dict, relations_dict: dict):
    df = pd.read_csv(f"Datasets/{datasets[dataset]}/{split}.txt", 
                     sep='\t',
                     header=None,
                     names=['head', 'relation', 'tail'], 
                     dtype={'head': str, 'tail': str})
    
    if dataset == 'yago':
        return df
    desc_dict = get_description_dict(dataset)
    return (
        df.assign(**{
        'description': df['head'].apply(lambda x: get_desc(str(x), dataset,desc_dict)),
        'head': df['head'].apply(lambda x: entities_dict.get(x, None)),
        'tail': df['tail'].apply(lambda x: entities_dict.get(x, None)),
        'relation':  df['relation'].apply(lambda x: relations_dict.get(str(x), None))
        }))


def build_graph(dataset: str, splits: list[str], entities_dict: dict, relations_dict: dict):
    nodes = set()
    graph = {}
    for split in splits:
        with open(f"Datasets/{datasets[dataset]}/{split}.txt", "r") as file:
            for line in file:
                node1, relation, node2 = line.strip().split('\t')
                
                if dataset=='fb15k':
                    relation = shorten_relation(relation)
                    relation = fb15k_mapping[relation]
                    node1, relation, node2 = (
                        entities_dict[node1],
                        relation,
                        entities_dict[node2]
                        )
                    
                if dataset not in ['yago', 'fb15k']:
                    node1, relation, node2 = (
                        entities_dict[node1],
                        relations_dict.get(relation, None),
                        entities_dict[node2]
                        )
                    
                if relation:
                    nodes.add(node1)
                    # Check if the first node already exists in the dictionary
                    if node1 not in graph:
                        # If not, create a new dictionary for the node
                        graph[node1] = {}
                    # Add the neighboring node and the relationship to the dictionary for node1
                    graph[node1][node2] = relation
                    
        node_list = list(nodes)
        return graph, node_list



def generate_prompt(row, with_desc=True, with_context=True):
    prompt_parts = []
    # sentence = (
    #     f"Predict with the help of the given context and related facts, the tail entity [mask] by filling in the sentence."
    #     f"Return only the tail entity."
    #     f"Sentence: Head: {row['head']}, Relation: {row['relation']}, Tail: [mask]"
    # )
    sentence = (
         f"Sentence: Head: {row['head']}, Relation: {row['relation']}, Tail: [mask]"
     )
    prompt_parts.append(sentence)

    # Desc part
    if with_desc and 'description' in row and row['description']:
        description_part = f"Context about {row['head']}: {row['description']}"
        prompt_parts.append(description_part)

    # Context part
    if with_context and 'triplets' in row and row['triplets']:
        if len(row['triplets']) == 2:
            triples_formatted = "; ".join(
                f"{head}: {relation} {tail}" for (head, relation, tail) in row['triplets']
            )
        else:
            triples_formatted = (
                f"{row['triplets'][0]}: {row['triplets'][1]} {row['triplets'][2]}"
            )
        context_part = f"Following are some related facts: {triples_formatted}"
        prompt_parts.append(context_part)

    elif with_context and row['neighbor_fact']:
        context_part = (
            f"Following is an example of a neighbor fact with the same relation: "
            f"{row['neighbor_fact'][0]}: {row['neighbor_fact'][1]} {row['neighbor_fact'][2]}"
        )
        prompt_parts.append(context_part)

    elif with_context and row['same_relation_fact']:
        context_part = (
            f"Following is an example of a fact with the same relation: "
            f"{row['same_relation_fact'][0]}: {row['same_relation_fact'][1]} {row['same_relation_fact'][2]}"
        )
        prompt_parts.append(context_part)

    return "\n".join(prompt_parts)




def get_triplet(df_row, patterns, graph, seed=0):
    random.seed(seed)
    head, relation = df_row['head'], df_row['relation']
    
    triplet_pairs = []
    single_triplets = []
    # Check if the relation is part of a known composition pattern
    for pattern in patterns:
        if relation in pattern:
            position = pattern.index(relation) + 1  # Position of relation in pattern (1, 2, or 3)

            if position == 3:
                # Retrieve triplets for position 3 pattern: (_, _, relation)
                first_relation, second_relation = pattern[0], pattern[1]
                

                # Find all neighbors for the first relation
                for neighbor, rel in graph.get(head, {}).items():
                    if rel == first_relation:
                        single_triplets.append((head, first_relation, neighbor))
                       
                        # Check if this intermediate node has a second_relation to another entity
                        for next_neighbor, next_rel in graph.get(neighbor, {}).items():
                            if next_rel == second_relation:
                                triplet2 = (neighbor, second_relation, next_neighbor)
                                triplet_pairs.append(((head, first_relation, neighbor), triplet2))

    if triplet_pairs:
        return random.choice(triplet_pairs)
    elif single_triplets:
        return random.choice(single_triplets)


    return None



def number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return (
        f"trainable model parameters: {trainable_model_params}\n"
        f"all model parameters: {all_model_params}\n"
        f"percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
    )



def find_neighbor_with_same_relation(graph, head, relation):
    if head not in graph:
        return None

    for neighbor_tail in graph[head].keys():
        # Check if the relation from head to tail matches the given relation
        if neighbor_tail in graph:
            # Look for the relation in neighbor_tail's connections
            for tail, rel in graph[neighbor_tail].items():
                if rel == relation:
                    return (neighbor_tail, rel, tail)
    return None


  
def find_triplet_with_same_relation(graph, head, relation):
    for current_head in graph.keys():
        if current_head == head:
            continue
        for neighbor_tail, rel in graph[current_head].items():
            # Check if the relation from current_head to neighbor_tail matches the given relation
            if rel == relation:
                return (current_head, rel, neighbor_tail)
                
    return None