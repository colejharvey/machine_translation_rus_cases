# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## Working directory
import os

os.getcwd()

os.chdir('Documents')
os.chdir('Research projects')
os.chdir('machine_translation_rus_cases')
os.chdir('Translations')  #This is where saved texts go


## For counting tokens
import nltk

## Translation code

from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")


translation = pipeline("translation_ru_to_en", model=model, tokenizer=tokenizer)
## Loading the list of all json files

path = 'C:/Users/colharv/Documents/Research projects/machine_translation_rus_cases/Russian texts/Text_lists' #This is vhere json files come from
files = os.listdir(path)


## Loading a json file of sentences and translating in a loop
import pandas as pd

for i in range(167, 200, 1):    #Full version would say range(len(files)); current version is for testing
    filepath_current = [path, files[i]]
    filepath_current = '/'.join(filepath_current)
    sentence_data = pd.read_json(filepath_current)
    sentence_data.head

    ## Convert to list

    sentence_list = sentence_data.iloc[:, 0].tolist()
  
    
    ## Selecting the text in a for loop

  
    #Getting caseid
    filename_current = files[i]
    caseid = str ( ''.join(filter(str.isdigit, filename_current) ) )
    
    translated_text = list()  #Empty list for translations

    for j in range(len(sentence_list)):
        text = sentence_list[j]
        nltk_tokens = nltk.word_tokenize(text)  #Gets approx number of tokens
        if len(nltk_tokens) > 500:  #If a large sentence, follow this path to break up by semi-colons
            print("Long string detected; splitting into clauses...")    
            text_split = text.split(';')
          #  translated_longsent = list() #Empty list for long sentence clauses
            for q in range(len(text_split)):
                clause = text_split[q]
                translated_clause = translation(clause, max_length = 512)[0]['translation_text']
            #    translated_longsent.append(translated_clause)
            #' '.join(translated_longsent) #Joins list items together with a space in between
                translated_text.append(translated_clause)
        else:
            translated_sentence = translation(text, max_length = 512)[0]['translation_text']
            translated_text.append(translated_sentence)
            print("Current translation:", str(j), "of", str(len(sentence_list)))
        
    full_text = ' '.join(translated_text) #Joins list items together with a space in between
    full_text = full_text.replace(" &apos; ", "' ")  #This fixes an error with apostrophes
    print(full_text)

    ## Write to file
    filename_save = ["caseid", caseid] #The string "caseid" plus the actual id number
    filename_save = '_'.join(filename_save)
    filename_save = [filename_save, ".txt"]
    filename_save = ''.join(filename_save)  #I assume there is a less hackish way to do this...

    with open(filename_save,"w", encoding="utf-8") as f:
        f.write(full_text)
    print("Document", str(i), "completed.")    

#Html_file= open("case_1-1125_24_july_2020.html","w", encoding="utf-8")
#Html_file.write(full_text)
#Html_file.close()

# Checking how many tokens
#model_inputs = tokenizer(text, return_tensors = "pt")

#model_inputs



    
    
