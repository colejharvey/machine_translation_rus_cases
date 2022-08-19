# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## Working directory
import os

os.getcwd()

os.chdir('.spyder-py3')
os.chdir('machine_translate_ru_en')
os.chdir('machine_translation_rus_cases')
os.chdir('Translations')  #This is where saved texts go


## Translation code

from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")


translation = pipeline("translation_ru_to_en", model=model, tokenizer=tokenizer)
## Loading the list of all json files

path = 'C:/Users/colej/Documents/Research projects/scrape_russian_courts/Data/Text_lists' #This is vhere json files come from
files = os.listdir(path)


## Loading a json file of sentences and translating in a loop
import pandas as pd

for i in range(31, 40, 1):    #Full version would say range(len(files)); current version is for testing
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
        translated_sentence = translation(text, max_length = 512)[0]['translation_text']
        translated_text.append(translated_sentence)
        print("Current translation:", str(j), "of", str(len(sentence_list)))
        
    full_text = ' '.join(translated_text) #Joins list items together with a space in between
    print(full_text)

    ## Write to file
    filename_save = ["caseid", caseid] #The string "caseid" plus the actual id number
    filename_save = '_'.join(filename_save)
    filename_save = [filename_save, ".txt"]
    filename_save = ''.join(filename_save)  #I assume there is a less hackish way to do this...

    with open(filename_save,"w", encoding="utf-8") as f:
        f.write(full_text)


#Html_file= open("case_1-1125_24_july_2020.html","w", encoding="utf-8")
#Html_file.write(full_text)
#Html_file.close()

# Checking how many tokens
#model_inputs = tokenizer(text, return_tensors = "pt")

#model_inputs



    
    
