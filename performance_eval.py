import json
import pandas as pd
import numpy as np

#======================================================== Auxiliary Functions ========================================================
def conditional_prob(tokens, tokens_c, tokens_nc):
    prob = []
    for i in range(len(tokens)):
        nominator = sum(tokens_nc[i:])
        denominator = sum(tokens_nc[i:])+sum(tokens_c[i:])
        prob.append(nominator/denominator)
    return prob

def value_to_color(val, palette, color_min, color_max, n_colors):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min)
            val_position = min(max(val_position, 0), 1)
            ind = int(val_position * (n_colors - 1))
            return palette[ind]
        
def value_to_progress(val, size_min, size_max):
    return (val - size_min) / (size_max - size_min)

def parse_report(report):
    parts = report.split("## ")
    data = {}
    
    for part in parts[1:]:  
        lines = part.strip().split("\n")
        title = lines[0].strip() 
        content = "\n".join(lines[1:]).strip()  
        
        if title == "Justification":
            data[title] = content
        else:
            data[title] = lines[1].strip() if len(lines) > 1 else ''
    
    return data
    
def parse_domain(domain_tree):
    #Check if domain_tree is not a string (i.e., it might be NaN)
    if not isinstance(domain_tree, str):
        return domain_tree  
    #Otherwise, process the string
    parts = domain_tree.split("->")
    if len(parts) < 2:
        return np.nan  
    return parts[1].strip()

def get_dataframe(file):
    records = []
    with open(file, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            info = parse_report(json_obj['omni-judge'])
            if info == {}:
                continue
            try:
                correctness = info['Equivalence Judgement']
                if correctness == 'TRUE':
                    records.append({'domain': json_obj['domain'], 'difficulty': json_obj['difficulty'], 'problem': json_obj['problem'], 'answer': json_obj['answer'], 'omni-judge': json_obj['omni-judge'], 'correctness': True, 'completion_tokens': json_obj['completion_tokens']})
                else:
                    records.append({'domain': json_obj['domain'], 'difficulty': json_obj['difficulty'], 'problem': json_obj['problem'], 'answer': json_obj['answer'], 'omni-judge': json_obj['omni-judge'], 'correctness': False, 'completion_tokens': json_obj['completion_tokens']})
            except:
                continue
        
    Data_df = pd.DataFrame(records)
    return Data_df

def get_dataframe_reasoning_models(file):
    records = []
    with open(file, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            info = parse_report(json_obj['omni-judge'])
            if info == {}:
                continue
            try:
                correctness = info['Equivalence Judgement']
                if correctness == 'TRUE':
                    records.append({'domain': json_obj['domain'], 'difficulty': json_obj['difficulty'], 'problem': json_obj['problem'], 'answer': json_obj['answer'], 'omni-judge': json_obj['omni-judge'], 'correctness': True, 'reasoning_tokens': json_obj['reasoning_tokens']})
                else:
                    records.append({'domain': json_obj['domain'], 'difficulty': json_obj['difficulty'], 'problem': json_obj['problem'], 'answer': json_obj['answer'], 'omni-judge': json_obj['omni-judge'], 'correctness': False, 'reasoning_tokens': json_obj['reasoning_tokens']})
            except:
                continue
        
    Data_df = pd.DataFrame(records)
    return Data_df


#======================================================== Performance & Token Evaluation ========================================================
def total_performance(file):
    Data_df = get_dataframe(file)
    total_performance = {'correct': 0, 'total': 0, 'accuracy': 0}
    total_performance['total'] = len(Data_df)
    total_performance['correct'] = len(Data_df[Data_df['correctness'] == True])
    total_performance['accuracy'] = (total_performance['correct'] / total_performance['total']) * 100

    return total_performance

def total_tokens(file):
    Data_df = get_dataframe_reasoning_models(file)
    total_tokens = {'sum': 0, 'total': 0, 'avg': 0}
    total_tokens['total'] = len(Data_df)
    total_tokens['sum'] = sum(Data_df['reasoning_tokens'])
    total_tokens['avg'] = (sum(Data_df['reasoning_tokens']) / len(Data_df))

    return total_tokens

def domain_performance(file):
    Data_df = get_dataframe(file)

    #The multi-domain questions are taken into account for each domain by using explode
    Data_df = Data_df.explode('domain').reset_index()
    Data_df['domain'] = Data_df['domain'].apply(parse_domain)

    #Deduplicate the data as some multi-domain questions might have the same primary domain
    Data_df_deduplicated = Data_df.drop_duplicates(subset=None)
    Data_df_deduplicated = Data_df_deduplicated.dropna() #There is one empty domain tree that gets value nan by parse_domain

    #Join the calculus and precalculus domains
    Data_df_deduplicated['domain'] = Data_df_deduplicated['domain'].apply(lambda x: 'Calculus' if x == 'Precalculus' else x)

    Domains = Data_df_deduplicated['domain'].unique()
    domain_performance = {domain: {'correct': 0, 'total': 0, 'accuracy': 0} for domain in Domains}
    for domain in Domains:
        domain_df = Data_df_deduplicated[Data_df_deduplicated['domain'] == domain]
        domain_performance[domain]['total'] = len(domain_df)
        domain_performance[domain]['correct'] = len(domain_df[domain_df['correctness'] == True])
        domain_performance[domain]['accuracy'] = (domain_performance[domain]['correct'] / domain_performance[domain]['total']) * 100
 
    return domain_performance

def difficulty_performance(file):
    Data_df = get_dataframe(file)

    #Perform q-cut to divide the data into equally sized difficulty tiers
    Data_df['difficulty'] = pd.qcut(Data_df['difficulty'], 4, labels=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])

    difficulty_levels = Data_df['difficulty'].unique()
    difficulty_performance = {difficulty: {'correct': 0, 'total': 0, 'accuracy': 0} for difficulty in difficulty_levels}
    for difficulty in difficulty_levels:
        difficulty_df = Data_df[Data_df['difficulty'] == difficulty]
        difficulty_performance[difficulty]['total'] = len(difficulty_df)
        difficulty_performance[difficulty]['correct'] = len(difficulty_df[difficulty_df['correctness'] == True])
        difficulty_performance[difficulty]['accuracy'] = (difficulty_performance[difficulty]['correct'] / difficulty_performance[difficulty]['total']) * 100

    return difficulty_performance

def difficulty_tokens(file):
    Data_df = get_dataframe_reasoning_models(file)

    # Perform q-cut to divide the data into equally sized difficulty tiers
    Data_df['difficulty'] = pd.qcut(Data_df['difficulty'], 4, labels=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])

    difficulty_levels = Data_df['difficulty'].unique()
    difficulty_tokens = {difficulty: {'sum': 0, 'total': 0, 'avg': 0} for difficulty in difficulty_levels}
    for difficulty in difficulty_levels:
        difficulty_df = Data_df[Data_df['difficulty'] == difficulty]
        difficulty_tokens[difficulty]['total'] = len(difficulty_df)
        difficulty_tokens[difficulty]['sum'] = sum(difficulty_df['reasoning_tokens'])
        difficulty_tokens[difficulty]['avg'] = (sum(difficulty_df['reasoning_tokens']) / len(difficulty_df))

    return difficulty_tokens

def domain_per_difficulty_performance(file):
    Data_df = get_dataframe(file)

    #The multi-domain questions are taken into account for each domain by using explode
    Data_df = Data_df.explode('domain').reset_index()
    Data_df['domain'] = Data_df['domain'].apply(parse_domain)

    #Deduplicate the data as some multi-domain questions might have the same primary domain
    Data_df_deduplicated = Data_df.drop_duplicates(subset=None)
    Data_df_deduplicated = Data_df_deduplicated.dropna() #There is one empty domain tree that gets value nan by parse_domain

    #Join the calculus and precalculus domains
    Data_df_deduplicated['domain'] = Data_df_deduplicated['domain'].apply(lambda x: 'Calculus' if x == 'Precalculus' else x)

    #Perform q-cut to divide the data into equally sized difficulty tiers
    Data_df_deduplicated['difficulty'] = pd.qcut(Data_df_deduplicated['difficulty'], 4, labels=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])

    Domains = Data_df_deduplicated['domain'].unique()
    difficulty_levels = Data_df_deduplicated['difficulty'].unique()

    domain_per_difficulty_performance = {difficulty: {domain: {'correct': 0, 'total': 0, 'accuracy': 0} for domain in Domains} for difficulty in difficulty_levels}
    for difficulty in difficulty_levels:
        difficulty_df = Data_df_deduplicated[Data_df_deduplicated['difficulty'] == difficulty]
        for domain in Domains:
            domain_df = difficulty_df[difficulty_df['domain'] == domain]
            domain_per_difficulty_performance[difficulty][domain]['total'] = len(domain_df)
            domain_per_difficulty_performance[difficulty][domain]['correct'] = len(domain_df[domain_df['correctness'] == True])
            if len(domain_df) == 0:
                domain_per_difficulty_performance[difficulty][domain]['accuracy'] = 0
            else:
                domain_per_difficulty_performance[difficulty][domain]['accuracy'] = (len(domain_df[domain_df['correctness'] == True]) / len(domain_df)) * 100

    return domain_per_difficulty_performance

def domain_per_difficulty_tokens(file):
    Data_df = get_dataframe_reasoning_models(file)
    
    #The multi-domain questions are taken into account for each domain by using explode
    Data_df = Data_df.explode('domain').reset_index()
    Data_df['domain'] = Data_df['domain'].apply(parse_domain)

    #Deduplicate the data as some multi-domain questions might have the same primary domain
    Data_df_deduplicated = Data_df.drop_duplicates(subset=None)
    Data_df_deduplicated = Data_df_deduplicated.dropna() # there is one empty domain tree that gets value nan by parse_domain

    #Join the calculus and precalculus domains
    Data_df_deduplicated['domain'] = Data_df_deduplicated['domain'].apply(lambda x: 'Calculus' if x == 'Precalculus' else x)

    #Perform q-cut to divide the data into equally sized difficulty tiers
    Data_df_deduplicated['difficulty'] = pd.qcut(Data_df_deduplicated['difficulty'], 4, labels=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])

    Domains = Data_df_deduplicated['domain'].unique()
    difficulty_levels = Data_df_deduplicated['difficulty'].unique()

    domain_per_difficulty_tokens = {difficulty: {domain: {'sum': 0, 'total': 0, 'avg': 0} for domain in Domains} for difficulty in difficulty_levels}
    for difficulty in difficulty_levels:
        difficulty_df = Data_df_deduplicated[Data_df_deduplicated['difficulty'] == difficulty]
        for domain in Domains:
            domain_df = difficulty_df[difficulty_df['domain'] == domain]
            domain_per_difficulty_tokens[difficulty][domain]['total'] = len(domain_df)
            domain_per_difficulty_tokens[difficulty][domain]['sum'] = sum(domain_df['reasoning_tokens'])
            if len(domain_df) == 0:
                domain_per_difficulty_tokens[difficulty][domain]['avg'] = 0
            else:
                domain_per_difficulty_tokens[difficulty][domain]['avg'] = (sum(domain_df['reasoning_tokens']) / len(domain_df))

    return domain_per_difficulty_tokens
