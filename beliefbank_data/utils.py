import json
import nltk 
import csv
import nltk  # $ pip install nltk
from nltk.stem import PorterStemmer
from nltk.corpus import cmudict  # >>> nltk.download('cmudict')
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
    for syllables in pronunciations.get(word, []):
        return syllables[0][-1].isdigit()  # use only the first one

connector = {
    "IsA": [" is ", " is not "], 
    "HasA": [" has ", " does not have "],
    "CapableOf": [" is able to " , " is not able to "],
    "MadeOf": [" is made of ", " is not made of "], 
    "HasProperty": [" has the property of being ", " does not have the property of being "],
    "HasPart": [" has ", " does not have "]
}

def generate_assertion(entity, relation, true):
    category, recipient = relation.split(",")
    # print(entity, relation, true)
    if (category == "IsA"):
        subjectArticle = "An " if starts_with_vowel_sound(entity) else "A "
        objectArticle = "an " if starts_with_vowel_sound(recipient) else "a "
        sentence = subjectArticle + entity + (connector[category][0] if true else connector[category][1]) + objectArticle + recipient + "."

    elif (category in ["HasA", "MadeOf", "HasProperty", "CapableOf"]):
        subjectArticle = "An " if starts_with_vowel_sound(entity) else "A "
        sentence = subjectArticle + entity + (connector[category][0] if true else connector[category][1]) + recipient + "."

    elif (category  == "HasPart"):
        subjectArticle = "An " if starts_with_vowel_sound(entity) else "A "
        objectArticle = "an " if starts_with_vowel_sound(recipient) else "a "
        sentence = subjectArticle + entity + (connector[category][0] if true else connector[category][1]) + objectArticle + recipient + "."

    return sentence

question_connector = {
    "IsA": ["Is", ""], 
    "HasA": ["Does", " have a "],
    "CapableOf": ["Is", " capable of "],
    "MadeOf": ["Is", " made of "], 
    "HasProperty": ["Does", " have the property of being "],
    "HasPart": ["Does", " have a "]
}

def generate_question(entity, relation, true):
    category, recipient = relation.split(",")
    
    subjectArticle = " an " if starts_with_vowel_sound(entity) else " a "
    objectArticle = " an " if starts_with_vowel_sound(recipient) else " a "

    answer = "Yes" if true else "No"

    question = question_connector[category][0] + subjectArticle + entity + question_connector[category][1] + \
        (objectArticle if category in ["IsA"] else "") + recipient + "?" # + "|" + answer

    return question, answer

def generate_question_with_context(contextEntity, contextRelation, contextTrue, questionEntity, questionRelation, questionTrue):
    question, _ = generate_question(questionEntity, questionRelation, questionTrue)
    return generate_assertion(contextEntity, contextRelation, contextTrue) + " " + question


def generate_inverse_question(entity, relation, true):
    category, recipient = relation.split(",")
    
    subjectArticle = " an " if starts_with_vowel_sound(entity) else " a "
    objectArticle = " an " if starts_with_vowel_sound(recipient) else " a "

    answer = "Yes" if true else "No"
    question = question_connector[category][0] + subjectArticle + entity + question_connector[category][1] + " not" + \
        (objectArticle if category in ["IsA"] else "") + recipient + "?" # + "|" + answer

    return question, answer

# TODO: add depth for deeper iteration of constraints tree
def find_constraints(constraints, filter_dict={}, max_constraints=None):
    def filter_fn(link):
        for k, v in filter_dict.items():
            if link[k] != v:
                return False
        return True
    
    results = []
    for link in constraints['links']:
        if max_constraints is not None and len(results) > max_constraints:
            break
        if filter_fn(link):
            results.append(link)
    return results