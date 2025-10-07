from seqeval.metrics import classification_report
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import datasets
import torch
from torch.utils.data import DataLoader
from functools import partial
import re
import sys
import numpy as np
sys.path.append('../')
from utils import *
    

#relations = ['product_or_material_produced','manufacturer','distributed_by','industry','position_held','original_broadcaster','owned_by','founded_by','distribution_format',   'headquarters_location','stock_exchange','currency','parent_organization','chief_executive_officer','director_/_manager','owner_of','operator','member_of','employer','chairperson' 'platform','subsidiary','legal_form','publisher','developer','brand','business_division','location_of_formation','creator',]

relations = ['acquired_by','brand_of', 'client_of', 'collaboration', 'competitor_of', 'merged_with', 'product_or_service_of', 'regulated_by', 'shareholder_of', 'subsidiary_of', 'traded_on', 'undefined']
#relations =['RESIDES_IN', 'IS_OF_SIZE', 'IS_BORN_ON', 'CREATED', 'HAS_CONSEQUENCE', 'HAS_FOR_LENGTH', 'DIED_IN', 'START_DATE', 'INITIATED', 'HAS_CATEGORY', 'HAS_LATITUDE', 'GENDER_FEMALE', 'DEATHS_NUMBER', 'GENDER_MALE', 'IS_PART_OF', 'WEIGHS', 'IS_REGISTERED_AS', 'HAS_QUANTITY', 'IS_OF_NATIONALITY', 'INJURED_NUMBER', 'END_DATE', 'HAS_CONTROL_OVER', 'IS_COOPERATING_WITH', 'IS_BORN_IN', 'HAS_FOR_WIDTH', 'IS_AT_ODDS_WITH', 'HAS_COLOR', 'HAS_FAMILY_RELATIONSHIP', 'WAS_DISSOLVED_IN', 'HAS_FOR_HEIGHT', 'IS_DEAD_ON', 'STARTED_IN', 'OPERATES_IN', 'IS_LOCATED_IN', 'WAS_CREATED_IN', 'HAS_LONGITUDE', 'IS_IN_CONTACT_WITH']
def cvt_text_to_pred(ref, text):
    
    preds = []
    for pred_txt in text.strip('.').split(';'):
        pred_match = re.match(r'^(.*):(.*),(.*)$', pred_txt)
        if pred_match is not None:
            relation, word1, word2 = pred_match.group(1).strip(), pred_match.group(2).strip(), pred_match.group(3).strip()
            if relation in relations and word1 in ref and word2 in ref:
                preds.append((relation, word1, word2))
            else:
                print("Not found Error: ", relation, word1, word2, ref)    
        else:
            print("Parse Error: ", pred_txt)
            
    return preds


def map_output(feature):

    ref = feature['input']
    label = cvt_text_to_pred(ref, feature['output'])
    pred = cvt_text_to_pred(ref, feature['out_text'])
    
    return {'label': label, 'pred': pred}


def calc_metric(gt_list, pred_list):
    # Initialize variables for true positives, false positives, and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for (ground_truth, predicted_relations) in zip(gt_list, pred_list):
        # Calculate true positives, false positives, and false negatives
        for relation in predicted_relations:
            if relation in ground_truth:
                true_positives += 1
            else:
                false_positives += 1

        for relation in ground_truth:
            if relation not in predicted_relations:
                false_negatives += 1

    # Calculate precision, recall, and F1-Score
    if true_positives + false_positives ==0:
        precision =0
    else:
        precision = true_positives / (true_positives + false_positives)
    if true_positives + false_negatives ==0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)
    if precision + recall ==0:
        f1_score=0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    # Print the results
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1_score)
    
"""
def test_re(args, model, tokenizer):

    dataset = load_from_disk('/projects/melodi/mettaleb/FinGPT/fingpt/FinGPT_Benchmark/data/fingpt-finred')['test']#.select(range(50))
    dataset = dataset.train_test_split(0.999999, seed=42)['test']
    #dataset = dataset['test']
    dataset = dataset.map(partial(test_mapping, args), load_from_cache_file=False)
    
    def collate_fn(batch):
        inputs = tokenizer(
            [f["prompt"] for f in batch], return_tensors='pt',
            padding=True, max_length=args.max_length,
            return_token_type_ids=False
        )
        return inputs
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    out_text_list = []
    log_interval = len(dataloader) // 5

    for idx, inputs in enumerate(tqdm(dataloader)):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        res = model.generate(**inputs, max_length=args.max_length, eos_token_id=tokenizer.eos_token_id, max_new_tokens=256)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        if (idx + 1) % log_interval == 0:
            tqdm.write(f'{idx}: {res_sentences[0]}')
        out_text = [o.split("Answer: ")[1] for o in res_sentences]
        out_text_list += out_text
        torch.cuda.empty_cache()
    
    dataset = dataset.add_column("out_text", out_text_list)
    dataset = dataset.map(map_output, load_from_cache_file=False)    
    dataset = dataset.to_pandas()
    
    print(dataset)
    dataset.to_csv('tmp.csv')
    
    label = [[tuple(t) for t in d.tolist()] for d in dataset['label']]
    pred = [[tuple(t) for t in d.tolist()] for d in dataset['pred']]
    
    label_re = [[t[0] for t in d.tolist()] for d in dataset['label']]
    pred_re = [[t[0] for t in d.tolist()] for d in dataset['pred']]
    
    calc_metric(label, pred)
    
    calc_metric(label_re, pred_re)

    return dataset
"""
def test_re(args, model, tokenizer):
    # Charger l'ensemble de données
    dataset = load_from_disk('/projects/melodi/mettaleb/FinGPT/fingpt/FinGPT_Benchmark/data/fingpt-finred')['test']
    
    # Sélectionner un seul échantillon pour l'entraînement
    train_dataset = dataset.select([0])  # Utiliser le premier exemple, par exemple
    
    # Utiliser tout le dataset pour les tests
    test_dataset = dataset.map(partial(test_mapping, args), load_from_cache_file=False)
    
    # Définir la fonction de chargement
    def collate_fn(batch):
        inputs = tokenizer(
            [f["prompt"] for f in batch], return_tensors='pt',
            padding=True, max_length=args.max_length,
            return_token_type_ids=False
        )
        return inputs
    
    # Charger tout le jeu de test dans le DataLoader
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    out_text_list = []
    log_interval = len(dataloader) // 5 if len(dataloader) >= 5 else 1

    for idx, inputs in enumerate(tqdm(dataloader)):
        #print("Inputs to model:", tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True))
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        res = model.generate(**inputs, max_length=args.max_length, eos_token_id=tokenizer.eos_token_id, max_new_tokens=256)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        print("Generated sentences:", res_sentences)
        if (idx + 1) % log_interval == 0:
            tqdm.write(f'{idx}: {res_sentences[0]}')
        out_text = [o.split("Answer: ")[1] for o in res_sentences]
        out_text_list += out_text
        torch.cuda.empty_cache()

    with open("out_text_list.sent", "w") as f:
        for rel in out_text_list:
            f.write(f"{rel}\n")

    # Ajouter les prédictions dans le dataset de test
    test_dataset = test_dataset.add_column("out_text", out_text_list)
    test_dataset = test_dataset.map(map_output, load_from_cache_file=False)    
    test_dataset = test_dataset.to_pandas()
    
    print(test_dataset)
    test_dataset.to_csv('tmp.csv' , escapechar='\\')
    
    # Préparer les données pour le calcul des métriques
    label = [[tuple(t) for t in d.tolist()] for d in test_dataset['label']]
    pred = [[tuple(t) for t in d.tolist()] for d in test_dataset['pred']]
    
    label_re = [[t[0] for t in d.tolist()] for d in test_dataset['label']]
    pred_re = [[t[0] for t in d.tolist()] for d in test_dataset['pred']]
    
    # Calculer les métriques
    calc_metric(label, pred)
    calc_metric(label_re, pred_re)

    return test_dataset
