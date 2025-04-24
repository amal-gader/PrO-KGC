
import ast
import json
from dotenv import load_dotenv
import os
import openai
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

load_dotenv()

api_key = os.getenv('UNI_API_KEY')
client = openai.OpenAI(api_key=api_key, base_url="https://llms-inference.innkube.fim.uni-passau.de")


openai_key = os.getenv('OPENAI_API_KEY')

client_openai= openai.OpenAI(
    api_key=openai_key
)


# instruction="""
# Given a relation pattern (r1, r2, r3) and two examples of paths using these relations, determine whether this represents a valid or invalid composition pattern. 
# A valid composition pattern means that fact 1 and fact 2 necessarily imply fact 3. 
# Note that the relation 'derivationally related form' means in this KG that the terms are semantically related.
# Return only the final classification as follows: Valid/Invalid.
# Examples:
# Pattern: "('hypernym', 'hypernym', 'hypernym')": 
#     Example1:
#     Fact1: ("variation", "hypernym", "perturbation"),
#     Fact2: ("perturbation", "hypernym", "activity"),
#     Fact3: ("variation", "hypernym", "activity"),
#     Example2: 
#     Fact1: ("write", "hypernym", "create verbally")
#     Fact2: ("create verbally", "hypernym", "make")
#     Fact3: ("write", "hypernym", "make")
# Response: valid. Explanation: if A is a hypernym of B and B is a hypernym of C, then A is a hypernym of C.
# """


instruction="""
Given a relation pattern (r1, r2, r3) and two examples of paths using these relations, determine whether this represents a valid or invalid composition pattern. 
A valid composition pattern means that fact 1 and fact 2 necessarily imply fact 3. If this implication does not universally hold, classify it as invalid. 
Return only the final classification as follows: Valid/Invalid. 

Example:
Pattern: ("'olympic_medal_honor/olympics'", "'olympic_medal_honor/medal'", "'olympic_medal_honor/medal'")
  Example 1:
    Fact 1: ("'Dominican Republic'", "'olympic_medal_honor/olympics'", "'1984 Summer Olympics'")
    Fact 2: ("'1984 Summer Olympics'", "'olympic_medal_honor/medal'", "'Silver medal'")
    Fact 3: ("'Dominican Republic'", "'olympic_medal_honor/medal'", "'Silver medal'")
  Example 2:
    Fact 1: ("'Dominican Republic'", "'olympic_medal_honor/olympics'", "'1984 Summer Olympics'")
    Fact 2: ("'1984 Summer Olympics'", "'olympic_medal_honor/medal'", "'Bronze medal'")
    Fact 3: ("'Dominican Republic'", "'olympic_medal_honor/medal'", "'Bronze medal'")
Response: invalid Explanation: if the  '1984 Summer Olympics' has a Silver medal as honor and Dominican Republic won a medal doesn't necessarily imply that it won the Silver one

"""

# instruction="""
# Given a relation pattern (r1, r2, r3) and two examples of paths using these relations, determine whether this represents a valid or invalid composition pattern. 
# A valid composition pattern means that fact 1 and fact 2 necessarily imply fact 3. If this implication does not universally hold or if there are exceptions, classify it as invalid. 
# If you are uncertain or cannot definitively prove that fact 3 follows from fact 1 and fact 2, classify the pattern as invalid.
# Your response should be in the following format: Valid/Invalid. Explanation:

# Pattern ('diplomatic relation', 'member of', 'member of'):
# Example 1:
#   Fact 1: ('Senegal', 'diplomatic relation', 'Germany')
#   Fact 2: ('Germany', 'member of', 'International Telecommunication Union')
#   Fact 3: ('Senegal', 'member of', 'International Telecommunication Union')
# Example 2:
#   Fact 1: ('Senegal', 'diplomatic relation', 'Germany')
#   Fact 2: ('Germany', 'member of', 'African Development Bank')
#   Fact 3: ('Senegal', 'member of', 'African Development Bank')


# Pattern ('diplomatic relation', 'diplomatic relation', 'diplomatic relation'):
#    Example1:
#      Fact 1: ['Senegal', 'diplomatic relation', 'Germany'],
#      Fact 2: ['Germany', 'diplomatic relation', 'Australia'],
#      Fact 3: ['Senegal', 'diplomatic relation', 'Australia'],
#    Example2:
#      Fact 1: ['Senegal', 'diplomatic relation', 'Germany'],
#      Fact 2: ['Germany', 'diplomatic relation', ""People's Republic of China""],
#      Fact 3: ['Senegal', 'diplomatic relation', ""People's Republic of China""]
  
# Response: invalid. Explanation: if a country has a diplomatic relation with another that is a member of an organization doesn't imply that it's also part of that organization.
 
#  """

def inference(prompt: str, model='gpt-4-turbo'):
    
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


def format_prompt(pattern, examples):
    formatted = f"Pattern: {pattern}\n"
    for idx, example in enumerate(examples, 1):
        formatted += f"   Example{idx}:\n"
        for fact_idx, fact in enumerate(example, 1):
            formatted += f"     Fact {fact_idx}: {fact},\n"
    return formatted.strip()  





def validate_patterns(file_path, dataset, model):
    #file_path = "lookup_files/patterns_codex-m_dict.json"
    with open(file_path, "r") as file:
        patterns = json.load(file)
    data = []
    for pattern, examples in patterns.items():
        #normalized_pattern = normalize_pattern(pattern)
        prompt = format_prompt(pattern, examples)
        data.append((pattern, prompt))


    df = pd.DataFrame(data, columns=["Pattern", "Prompt"])

    df['label'] = df['Prompt'].progress_apply(lambda x: inference(x))
    df.to_csv(f'validated_patterns_{dataset}_{model}.tsv', sep='\t', encoding='utf-8', index=False)





def filter_patterns(file_path: str):
    #file_path="validated_patterns_fb15k_gpt"
    patterns_1= pd.read_csv(f"{file_path}.tsv", sep='\t')
    
    label_pattern = r"(?i)\b(valid|invalid)\b[\.,]?"

    #explanation_pattern = r"Explanation:\s*(.+)"
    patterns_1['label (valid/invalid)']=patterns_1['label'].str.extract(label_pattern)

    labels = pd.concat(
        [patterns_1['pattern'],patterns_1['Prompt'],
        patterns_1['label (valid/invalid)'], 
        ], 
        axis=1
    )

    labels['pattern']=labels['pattern'].apply(ast.literal_eval)
    labels[['pattern', 'label (valid/invalid)']].to_csv(f"{file_path}.txt",header=None, index=False)




if __name__=='__main__':
    #filter_patterns()
    file_path="validated_patterns_fb15k_gpt"
    patterns_path = "patterns_fb15k_gpt.txt"
    
    patterns= pd.read_csv(file_path,header=None, sep='\t')
    label_pattern = r"(?i)\b(valid|invalid)\b[\.,]?"
    patterns['label (valid/invalid)']=patterns[0].str.extract(label_pattern)
    patterns[0] = patterns[0].str.replace(',Valid', '', regex=True)
     
    patterns[0].to_csv(patterns_path,header=None, index=False, sep='\t')
    print(patterns.iloc[1][0])