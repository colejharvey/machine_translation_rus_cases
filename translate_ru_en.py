# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## Loading a json file of sentences
import pandas as pd

sentence_data = pd.read_json("C:/Users/colej/Documents/Research projects/scrape_russian_courts/Data/Text_lists/test.json", encoding='UTF-8')
sentence_data.head

## Convert to list

sentence_list = sentence_data.iloc[:, 0].tolist()
  
    
## Translation code

from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")


translation = pipeline("translation_ru_to_en", model=model, tokenizer=tokenizer)

## Selecting the text in a for loop

#sentence_data.set_index("text1")
#sentence_data.loc[[1]]

translated_text = list()

for i in range(len(sentence_list)):
    text = sentence_list[i]
    translated_sentence = translation(text, max_length = 512)[0]['translation_text']
    translated_text.append(translated_sentence)    

full_text = ''.join(translated_text)
print(full_text)


#Checking how many tokens
model_inputs = tokenizer(text, return_tensors = "pt")

model_inputs



    len(sentence_list)
    
    
