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


        
f = open("calibration_facts.json")
data = json.load(f)

possibilities = []
for i in data:
    for property in data[i]:
        if ("CapableOf" in property and "ing" in property):
            print(property)
        possibilities.append(property.split(",")[0])

print(set(possibilities))

#{'IsA', 'HasProperty', 'MadeOf', 'HasA', 'HasPart', 'CapableOf'}


connector = {"IsA": [" is ", " is not "], 
             "HasA": [" has ", " does not have "],
            "CapableOf": [" is able to " , " is not able to "],
            "MadeOf": [" is made of ", " is not made of "], 
            "HasProperty": [" has the property of being ", " does not have the property of being "],
            "HasPart": [" has ", " does not have "]
            }

### Generate questions
with open('calibration_sentences.csv', 'w', newline='') as csvfile:
    for i in data:
        for property in data[i]:
            true = (data[i][property] == "yes")
            category, recipient = property.split(",")
            #print(true, category, recipient)

            if (category == "IsA"):
                subjectArticle = "An " if starts_with_vowel_sound(i) else "A "
                objectArticle = "an " if starts_with_vowel_sound(recipient) else "a "
                sentence = subjectArticle + i + (connector[category][0] if true else connector[category][1]) + objectArticle + recipient + "."
                
            elif (category in ["HasA", "MadeOf", "HasProperty", "CapableOf"]):
                subjectArticle = "An " if starts_with_vowel_sound(i) else "A "
                sentence = subjectArticle + i + (connector[category][0] if true else connector[category][1]) + recipient + "."
                
            elif (category  == "HasPart"):
                subjectArticle = "An " if starts_with_vowel_sound(i) else "A "
                objectArticle = "an " if starts_with_vowel_sound(recipient) else "a "
                sentence = subjectArticle + i + (connector[category][0] if true else connector[category][1]) + objectArticle + recipient + "."

            print(sentence)
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([str(sentence)])

question_connector = {"IsA": ["Is", ""], 
             "HasA": ["Does", " have a "],
            "CapableOf": ["Is", " capable of "],
            "MadeOf": ["Is", " made of "], 
            "HasProperty": ["Does", " have the property of being "],
            "HasPart": ["Does", " have a "]
            }

### Generate Questions and Answers
with open('calibration_questions.csv', 'w', newline='') as csvfile:
    for i in data:
        for property in data[i]:
            true = (data[i][property] == "yes")
            category, recipient = property.split(",")

            subjectArticle = " an " if starts_with_vowel_sound(i) else " a "
            objectArticle = " an " if starts_with_vowel_sound(recipient) else " a "

            answer = "Yes" if true else "No"
            question = question_connector[category][0] + subjectArticle + i + question_connector[category][1] + \
                (objectArticle if category in ["IsA"] else "") + recipient + "? " + answer

            print(question)
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([str(question)])