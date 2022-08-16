# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:07:56 2022

@author: colej
"""

# Importing the mBART functions from transformer library
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="ru_RU")


text = "Назначить Брагину Андрею Михайловичу, Буркевицу Алексею Александровичу, каждому, меру уголовно-правового характера в виде судебного штрафа в размере 50 000 рублей, который подлежит уплате в доход федерального бюджета в течение 90 дней со дня вступления настоящего постановления в законную силу.Штраф подлежит уплате по следующим реквизитам:УФК по Курганской области (СУ СК России по Курганской области л/с №)ИНН/КПП №БИК №р/с № № в отделении КурганОКТМО №КБК №.Разъяснить Брагину А.М., Буркевицу А.А., что сведения об уплате судебного штрафа должны быть представлены каждым из них судебному приставу-исполнителю не позднее 10 дней после истечения срока, установленного для уплаты судебного штрафа, а неуплата судебного штрафа в установленный судом срок согласно ч. 2 ст. 104.4 УК РФ, ст. 446.5 УПК РФ является основанием для отмены настоящего постановления и привлечения их к уголовной ответственности.До вступления постановления в законную силу избранные в отношении Брагина А.М., Буркевица А.А. меры пресечения в виде домашнего ареста отменить."


model_inputs = tokenizer(text, return_tensors = "pt")


# Russian to English translation
generated_tokens = model.generate(**model_inputs, forced_bos_token_id = tokenizer.lang_code_to_id ["en_XX"])


translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
translation