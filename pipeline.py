import os, sys
import random
import json
import nltk 
import csv
import torch
import numpy as np
import nltk  # $ pip install nltk
from nltk.stem import PorterStemmer
from nltk.corpus import cmudict  # >>> nltk.download('cmudict')
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

from beliefbank_data.utils import generate_assertion, generate_question, find_constraints

"""Load data"""

constraints_path = "beliefbank_data/constraints_v2.json"
facts_path = "beliefbank_data/silver_facts.json"
constraints = json.load(open(constraints_path))
facts = json.load(open(facts_path))

with open("beliefbank_data/dev_entities.txt", "r") as f:
    dev_entities = [e.strip() for e in f.readlines()]
    
statements = [(entity, relation, label == 'yes')
              for entity, relations in facts.items() if entity in dev_entities 
              for relation, label in relations.items()]
print(f"Number of facts: {len(statements)}")

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

def predict(question_list):
    B = len(question_list)
    question_list = format_question(question_list)
    answer_list_all_yes = ["$answer$ = yes"] * B     # pass in list of "yes"
    
    # print(dir(tokenizer))
    inputs = tokenizer.batch_encode_plus(question_list, max_length = 256, padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer.batch_encode_plus(answer_list_all_yes, max_length = 15, padding=True, truncation=True, return_tensors="pt") # max_length is set to len("$answer$ = yes")
    
    fwd = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                labels=labels["input_ids"])

    logits = fwd.logits.reshape((B, 7, -1))
    logits = logits[:, 5, :] # Index of yes/no token in answer
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    # yes has input_id 4273, no has input_id 150
    confidence_yes = probs[..., 4273] 
    confidence_no = probs[..., 150]
    
    answers = (confidence_yes >= confidence_no) # np.array([(ans == "$answer$ = yes") for ans in answers])
    confidences = np.where(answers, confidence_yes, confidence_no)
    return answers, confidences

# Load NLI model
print("Loading NLI model...")
nli_tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
nli_model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
nli_model = nli_model.to(device=device).eval()
print("NLI model loaded!")

# NLI model functions
def contradiction_matrix(sents, nli_tokenizer, nli_model):
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

    tokenized = nli_tokenizer(prem, hypo, 
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
    predicted_probability = torch.softmax(nli_outputs.logits, dim=1)
    contra_matrix = predicted_probability[..., 2]
    contra_matrix = contra_matrix.reshape(N, B, B)
    return contra_matrix.detach().cpu().numpy()

# Correction functions
def do_nothing(predictions, confidences, contra_matrix):
    return predictions

def correction_1(predictions, confidences, contra_matrix):
    contra_matrix_sym = (contra_matrix + contra_matrix.T) / 2
    pass
    return predictions

# Evaluation functions
def evaluate(predictions, answers):
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    answers = answers.reshape(predictions.shape)
    
    actual_answers = answers.copy()
    actual_answers[:, 1:] = np.logical_not(actual_answers[:, 1:])
    acc = np.sum(predictions == actual_answers)
    
    yes_no = (answers[:, 0] == answers[:, 1])
    pred_same = (predictions[:, 0] == predictions[:, 1])
    pred_diff = np.logical_not(pred_same)
    con = np.where(yes_no, pred_same, pred_diff)
    con = np.count_nonzero(con)
    
    total = predictions.size
    bsize = predictions.shape[0]
    return acc, con, total, bsize

# Run experiment
print("Beginning experiment...")

acc_count = 0
con_count = 0
total_count = 0
num_pairs = 0

batch_size = 40
batch_counter = 0
num_batches = 0
batch = []
for idx, base in enumerate(statements):
    entity, relation, true = base
    
    filter_dict = {
        'source': relation,
        'direction': 'forward',
    }
    selected_constraints = find_constraints(constraints, filter_dict=filter_dict)
    if len(selected_constraints) == 0:
        continue
    c = random.choice(selected_constraints)
    contra = (entity, c['target'], not (c['weight'] == 'yes_yes'))
    # print(base, contra)
    
    # batch = [base, contra]
    if batch_counter == 0:
        batch = []
    batch_counter += 1
    batch.extend([base, contra])
    if batch_counter < batch_size: # Batch not full yet, keep accumulating examples
        continue
    # We have a full batch
    batch_counter = 0
    num_batches += 1
    
    questions, answers = zip(*[generate_question(*tup) for tup in batch])
    question_list = list(questions)
    answer_list = np.array([ans == "Yes" for ans in answers])
    # print("Questions:", question_list)
    # print("Labels (for contradiction):", answer_list)
    
    predictions, confidences = predict(question_list)
    predictions = predictions.flatten()
    confidences = confidences.flatten()
    # print("QA predictions:", predictions)
    # print("QA confidences:", confidences)
    
    pred_batch = [(ent, rel, predictions[i]) for i, (ent, rel, true) in enumerate(batch)]
    assertions = [generate_assertion(*tup) for tup in pred_batch]
    # print("Assertions:", assertions)
    
    assertions = np.array(assertions).reshape(batch_size, -1)
    contra_matrix = contradiction_matrix(assertions, nli_tokenizer, nli_model)
    # print("Contradiction probability matrix:\n", contra_matrix)
    
    predictions = predictions.reshape(batch_size, -1)
    confidences = confidences.reshape(batch_size, -1)
    corrected = do_nothing(predictions, confidences, contra_matrix)
    acc, con, total, bsize = evaluate(corrected, answer_list)
    acc_count += acc
    con_count += con
    total_count += total
    num_pairs += bsize
    # print(acc, con, total, bsize)
    
    if num_batches % 5 == 0:
        print(f"Iter {idx}: {num_batches} batches, {num_pairs} pairs")
        print(f"\tAccurate {acc_count} / {total_count} = {acc_count / total_count}")
        print(f"\tContradictions {con_count} / {num_pairs} = {con_count / num_pairs}")
    
print("\n======================== Final Report ========================")
print(f"{num_batches} batches, {num_pairs} pairs")
print(f"Accurate {acc_count} / {total_count} = {acc_count / total_count}")
print(f"Contradictions {con_count} / {total_count // 2} = {con_count / num_pairs}")