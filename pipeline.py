import os, sys
import random
import json
import nltk 
import csv
import time
import copy
import torch
import numpy as np
import nltk  # $ pip install nltk
from nltk.stem import PorterStemmer
from nltk.corpus import cmudict  # >>> nltk.download('cmudict')
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

from beliefbank_data.utils import generate_assertion, generate_question, find_constraints
import correction_utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_batches', required=False, type=int, default=10000, \
                    help='Number of batches to run. Defaults to 10000')
parser.add_argument('--bsize', required=False, type=int, default=10, \
                    help="Size of each batch of questions/facts. Defaults to 10")
parser.add_argument('--big_bsize', required=False, type=int, default=10, \
                    help="Number of batches to collect before performing calculations. Defaults to 10")
parser.add_argument('--max_bsize_qa', required=False, type=int, default=20, \
                   help="Maximum forward pass batch size for the QA model. Defaults to 20")
parser.add_argument('--max_bsize_nli', required=False, type=int, default=20, \
                   help="Maximum forward pass batch size for the NLI model. Defaults to 20")
parser.add_argument('--correction_fn', required=False, type=str, nargs='+', default=["do_nothing"], \
                   help="Function(s) from correction_utils. Defaults to \"do_nothing\"")

parser.add_argument('--constraints_path', required=False, type=str, default="beliefbank_data/constraints_v2.json", \
                   help="Path to constraints json file. Defaults to \"beliefbank_data/constraints_v2.json\"")
parser.add_argument('--facts_path', required=False, type=str, default="beliefbank_data/silver_facts.json", \
                    help="Path to facts json file. Defaults to \"beliefbank_data/silver_facts.json\"")
parser.add_argument('--entities_path', required=False, type=str, default="beliefbank_data/dev_entities.txt",
                   help="Path to text file listing entities of the current split. Defaults to \"beliefbank_data/dev_entities.txt\"")
parser.add_argument('--constraints_depth', required=False, type=int, default=10, \
                   help="Depth to extend constraints links, depth=1 corresponding to original constraints graph. Defaults to 10.")

args = parser.parse_args()

print(args)

"""Load data"""

constraints_path = args.constraints_path
facts_path = args.facts_path
constraints = json.load(open(constraints_path))
facts = json.load(open(facts_path))

with open(args.entities_path, "r") as f:
    dev_entities = [e.strip() for e in f.readlines()]
    
"""Constraint checking"""
constraints_yy = set() # A implies B
constraints_yn = set() # A implies not B
for link in constraints['links']:
    s = link['source']
    t = link['target']
    if link['weight'] == 'yes_yes':
        if link['direction'] == 'forward':
            constraints_yy.add((s, t))
        else:
            constraints_yy.add((t, s))
    else:
        constraints_yn.add((s, t))
        constraints_yn.add((t, s))

dict_yy = {}
for s, t in constraints_yy:
    if s in dict_yy:
        dict_yy[s].add(t)
    else:
        dict_yy[s] = {t}
dict_yn = {}
for s, t in constraints_yn:
    if s in dict_yn:
        dict_yn[s].add(t)
    else:
        dict_yn[s] = {t}
    
depth = args.constraints_depth
len_yy = [len(constraints_yy)]
len_yn = [len(constraints_yn)]
for d in range(depth-1):
    temp_yy = copy.deepcopy(dict_yy)
    temp_yn = copy.deepcopy(dict_yn)
    for a, bs in dict_yy.items():
        for b in bs:
            for c in dict_yy.get(b, set()):
                if a == c:
                    continue
                temp_yy[a].add(c)
            for c in dict_yn.get(b, set()):
                if a == c:
                    continue
                if a in temp_yn:
                    temp_yn[a].add(c)
                else:
                    temp_yn[a] = {c}
                if c in temp_yn:
                    temp_yn[c].add(a)
                else:
                    temp_yn[c] = {a}
    del dict_yy
    del dict_yn
    dict_yy = temp_yy
    dict_yn = temp_yn
    len_yy.append(sum([len(v) for v in dict_yy.values()]))
    len_yn.append(sum([len(v) for v in dict_yn.values()]))
    
for s, ts in dict_yy.items():
    for t in ts:
        constraints_yy.add((s, t))
for s, ts in dict_yn.items():
    for t in ts:
        constraints_yn.add((s, t))
constraints_nn = set([(t, s) for s, t in constraints_yy])
dict_nn = {}
for s, t in constraints_nn:
    if s in dict_nn:
        dict_nn[s].add(t)
    else:
        dict_nn[s] = {t}

print("yes -> yes constraints:", len(constraints_yy))
print("yes -> no constraints:", len(constraints_yn))
print("no -> no constraints:", len(constraints_nn))

neighbors = {}
num_neighbors = {}
for s in list(dict_yy.keys()) + list(dict_yn.keys()) + list(dict_nn.keys()):
    neighbors[s] = dict_yy.get(s, set()).union(dict_yn.get(s, set())).union(dict_nn.get(s, set()))
    num_neighbors[s] = len(neighbors[s])
    
def check_constraints(relation1, true1, relation2, true2):
    # Is (relation1, true1) & (relation2, true2)  
    if true1 and true2: # Case a
        implies12 = (relation1, relation2) in constraints_yy
        implies21 = (relation2, relation1) in constraints_yy
        contradicts = (relation1, relation2) in constraints_yn or (relation2, relation1) in constraints_yn
    elif true1 and not true2: # Case b
        implies12 = (relation1, relation2) in constraints_yn or (relation2, relation1) in constraints_yn
        implies21 = False
        contradicts = (relation1, relation2) in constraints_yy
    elif not true1 and true2: # Case c
        implies12 = False
        implies21 = (relation1, relation2) in constraints_yn or (relation2, relation1) in constraints_yn
        contradicts = (relation2, relation1) in constraints_yy
    else: # Case d
        implies12 = (relation2, relation1) in constraints_yy
        implies21 = (relation1, relation2) in constraints_yy
        contradicts = False
    return implies12, implies21, contradicts

"""Load models"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Working on device", device)

# Load QA model
print("Loading QA model...")
tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-large")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large")
model = model.to(device=device).eval()
print("QA model loaded!")

# QA model functions
def format_question(question_list):
    question_list = ["$answer$ ; $mcoptions$ = (A) yes (B) no; $question$ = " + item \
         for item in question_list]
    return question_list

def predict(question_list, max_bsize=10):
    B = len(question_list)
    question_list = format_question(question_list)
    answer_list_all_yes = ["$answer$ = yes"] * B     # pass in list of "yes"
    
    answers_all = []
    confidences_all = []
    for i in range(0, B, max_bsize):
        j = min(i + max_bsize, B)
        # print(dir(tokenizer))
        inputs = tokenizer.batch_encode_plus(question_list[i:j], max_length = 256, padding=True, truncation=True, return_tensors="pt")
        labels = tokenizer.batch_encode_plus(answer_list_all_yes[i:j], max_length = 15, padding=True, truncation=True, return_tensors="pt") # max_length is set to len("$answer$ = yes")

        # output = model.generate(input_ids, max_length=200)
        # answers = tokenizer.batch_decode(output, skip_special_tokens=True)
        fwd = model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device),
                    labels=labels["input_ids"].to(device))
                    # decoder_input_ids=labels["input_ids"], decoder_attention_mask=labels["attention_mask"])
        # output_ids = torch.argmax(fwd.logits, dim=-1)
        # print(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

        # loss
        # loss = fwd.loss # - log(P(y|x))
        # confidence = torch.exp(-loss)
        logits = fwd.logits.reshape((j - i, 7, -1))
        logits = logits[:, 5, :] # Index of yes/no token in answer
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        # yes has input_id 4273, no has input_id 150
        confidence_yes = probs[..., 4273] 
        confidence_no = probs[..., 150]

        answers = (confidence_yes >= confidence_no) # np.array([(ans == "$answer$ = yes") for ans in answers])
        confidences = np.where(answers, confidence_yes, confidence_no)
        answers_all.append(answers)
        confidences_all.append(confidences)
    answers = np.concatenate(answers_all, axis=0)
    confidences = np.concatenate(confidences_all, axis=0)
    return answers, confidences

# Load NLI model
print("Loading NLI model...")
nli_tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
nli_model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
nli_model = nli_model.to(device=device).eval()
print("NLI model loaded!")

# NLI model functions
def nli(sents, nli_tokenizer, nli_model, max_bsize=20):
    """Generates contradiction matrix of shape (N, B, B)"""
    if sents.ndim == 1:
        sents = sents.reshape(1, -1)
    
    N, B = sents.shape
    prem = []
    hypo = []
    for i in range(N):
        for j in range(B):
            for k in range(B):
                prem.append(sents[i][j])
                hypo.append(sents[i][k])

    nli_matrix = []
    size = N * B * B
    for i in range(0, size, max_bsize):
        j = min(i + max_bsize, size)
        tokenized = nli_tokenizer(prem[i:j], hypo[i:j], 
                                  max_length=256, 
                                  return_token_type_ids=True, 
                                  truncation=True,
                                  padding=True)

        input_ids = torch.Tensor(tokenized['input_ids']).to(device).long()
        token_type_ids = torch.Tensor(tokenized['token_type_ids']).to(device).long()
        attention_mask = torch.Tensor(tokenized['attention_mask']).to(device).long()

        nli_outputs = nli_model(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=None)
        nli_matrix.append(torch.softmax(nli_outputs.logits.detach().cpu(), dim=1))
    nli_matrix = torch.cat(nli_matrix, dim=0)
    nli_matrix = nli_matrix.reshape(N, B, B, 3)
    return nli_matrix.numpy()

# Correction functions
def do_nothing(predictions, confidences, contra_matrix):
    return predictions

def correction_1(predictions, confidences, contra_matrix):
    contra_matrix_sym = (contra_matrix + contra_matrix.T) / 2
    pass
    return predictions

# Evaluation functions
def evaluate(predictions, answers, pred_batch):
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    answers = answers.reshape(predictions.shape)
    relations = np.array([rel for ent, rel, pred in pred_batch]).reshape(predictions.shape)
    N, B = predictions.shape
    # predictions, answers, relations hould be size (N, B)
    
    # Calculate accurate examples
    acc = np.sum(predictions == answers)
    
    # Calculate contradictions
    con = 0
    for i in range(N):
        for j in range(B):
            for k in range(j+1, B):
                impl12, impl21, contra = check_constraints( \
                    relations[i, j], predictions[i, j], relations[i, k], predictions[i, k])
                if contra:
                    con += 1
    
    total = predictions.size
    bsize = predictions.shape[0]
    return acc, con, total, bsize

# Run experiment
print("Beginning experiment...")

start = time.time()

correction_fn_names = args.correction_fn
correction_fns = [getattr(correction_utils, fn_name) for fn_name in correction_fn_names]

acc_count = [0] * len(correction_fns)
con_count = [0] * len(correction_fns)
total_count = [0] * len(correction_fns)
num_batches = [0] * len(correction_fns)
flip_count = [0] * len(correction_fns)

N = args.big_bsize # Number of entities to sample in a big batch
batch_counter = 0
num_big_batches = 0
big_batch = []

B = args.bsize # Number of facts for each entity

random.shuffle(dev_entities)
idx_count = 0
idx = 0
while num_batches[0] < args.num_batches:
    if idx == len(dev_entities):
        random.shuffle(dev_entities)
        idx = 0
    entity = dev_entities[idx]
    idx += 1
    idx_count += 1
    
    # Sample set of facts for an entity
    # Sampling method 2
    entity_facts = list(facts[entity].items())
    base = random.choice(entity_facts)
    relation, label = base
    nearby = neighbors.get(relation, set())
    allowed_facts = [f for f in entity_facts if f[0] in nearby]
    if len(allowed_facts) < B - 1:
        continue
    batch = random.sample(allowed_facts, min(B - 1, len(allowed_facts)))
    batch = [base] + batch
    batch = [(entity, rel, label == "yes") for rel, label in batch]
    
    # # Random sampling (method 3)
    # batch = random.sample(list(facts[entity].items()), B)
    # batch = [(entity, relation, true == "yes") for relation, yes in batch]
    
    # Collect batches in big batches
    if batch_counter == 0:
        big_batch = []
    batch_counter += 1
    big_batch.extend(batch)
    if batch_counter < N: # Big batch not full yet, keep accumulating examples
        continue
    # We have a full batch
    batch_counter = 0
    num_big_batches += 1
    
    questions, answers = zip(*[generate_question(*tup) for tup in big_batch])
    questions = list(questions)
    answers = np.array([ans == "Yes" for ans in answers])
    # print("Questions:", question_list)
    # print("Labels (for contradiction):", answer_list)
    
    # Run through QA model
    predictions, confidences = predict(questions, max_bsize=args.max_bsize_qa)
    predictions = predictions.flatten()
    confidences = confidences.flatten()
    # print("QA predictions:", predictions)
    # print("QA confidences:", confidences)
    
    pred_batch = [(ent, rel, predictions[i]) for i, (ent, rel, true) in enumerate(big_batch)]
    assertions = [generate_assertion(*tup) for tup in pred_batch]
    # print("Assertions:", assertions)
    
    # Run through NLI model
    assertions = np.array(assertions).reshape(N, B)
    nli_matrix = nli(assertions, nli_tokenizer, nli_model, max_bsize=args.max_bsize_nli)
    # print("NLI probability matrix:\n", nli_matrix)
    
    predictions = predictions.reshape(N, B)
    confidences = confidences.reshape(N, B)
    
    for i, correction_fn in enumerate(correction_fns):
        corrected, flip_mask = correction_fn(predictions.copy(), confidences.copy(), nli_matrix.copy(), return_flip_mask=True)
        flip_count[i] += np.count_nonzero(flip_mask)
        acc, con, total, bsize = evaluate(corrected, answers, pred_batch)
        acc_count[i] += acc
        con_count[i] += con
        total_count[i] += total
        num_batches[i] += bsize # bsize should be equal to N
    # print(acc, con, total, bsize)
    
    if num_batches[0] % 50 == 0:
        num_pairs = num_batches[0] * B * B
        print(f"Iter {idx_count}: {num_batches[0]} batches, {total_count[0]} facts")
        for i, fn_name in enumerate(correction_fn_names):
            print(f"Correction function {fn_name}:")
            print(f"\tAccurate {acc_count[i]} / {total_count[i]} = {acc_count[i] / total_count[i]}")
            print(f"\tContradictions {con_count[i]} / {num_batches[i]} = {con_count[i] / num_batches[i]}")
            print(f"\tCorrections {flip_count[i]} / {num_batches[i]} = {flip_count[i] / num_batches[i]}")

print("\n==================== Final Results ====================")
num_pairs = num_batches[0] * B * B
print(f"End on iter {idx_count}: {num_batches[0]} {B}x{B} batches, {total_count[0]} facts")
for i, fn_name in enumerate(correction_fn_names):
    print(f"Correction function {fn_name}:")
    print(f"\tAccurate {acc_count[i]} / {total_count[i]} questions = {acc_count[i] / total_count[i]}")
    print(f"\tContradictions {con_count[i]} / {num_batches[i]} batches = {con_count[i] / num_batches[i]} contradictions per batch")
    print(f"\tCorrections {flip_count[i]} / {num_batches[i]} batches = {flip_count[i] / num_batches[i]} flips per batch")

end = time.time()
print("Runtime:", end - start)