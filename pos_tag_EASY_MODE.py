# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:18:48 2020

@author: sirre
"""
import numpy as np
import nltk 
import re
import pandas as pd

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def pos_sentence_constructor(pos_token_list):
    string = ''
    for i in range(len(pos_token_list)):
        string += '<' + str(pos_token_list[i])+'>' + ' '
    return string

def preprocessing_sentences(sentence):
    temp1 = re.sub(r'[^a-zA-z0-9]',' ',sentence)
    temp1 = temp1.lower()
    tokenized = nltk.word_tokenize(temp1)
    pos_tag_set = nltk.pos_tag(tokenized)
    return pos_tag_set

dict_pos = {'<conjuction>':['<CC>'],'<cardinal_digit>':['<CD>'],'<determiner>':['<DT>'],'<existential_there>':['<EX>'],'<foreign_word>':['<FW>'],
            '<preposition>':['<IN>'],'<adjective>':['<JJ>','<JJR>','<JJS>'],'<list_maker>':['<LS>'],'<modal>':['<MD>'],'<noun>':['<NN>','<NNS>','<NNP>','<NNPS>'],'<predeterminers>':['<PDT>','<POS>'],
            '<pronoun>':['<PRP>','PRP$'],'<adverb>':['<RB>','<RBR>','<RBS>','<RP>'],'<to>':['<TO>'],'<interjection>':['<UH>'],
            '<verb>':['<VB>','<VBD>','<VBN>','<VBP>','<VBZ>'],'<wh-word>':['<WDT>','<WP>','<WP$>','<WRB>']}
def get_key(val): 
    for key, value in dict_pos.items(): 
        for i in value:
         if val == i: 
             return key
    return '<UNK>'
         
def readable_pos(string):
    re_string = ''
    for words in string.split():
        keys = get_key(words) 
        re_string += str(keys) + ' '
    return re_string
    
#--------------driver code----------
sentence = """Miku, Miku, you can call me Miku. Blue hair, blue tie, hiding in your wi-fi. Open secrets, anyone can find me. Hear your music running through my mind
I'm thinking Miku, Miku. I'm thinking Miku, Miku).I'm thinking Miku, Miku.I'm thinking Miku, Miku.
I'm on top of the world because of you. All I wanted to do is follow you. I'll keep singing along to all of you. I'll keep singing along
I'm thinking Miku, Miku.I'm thinking Miku, Miku).I'm thinking Miku, Miku.I'm thinking Miku, Miku.
Miku, Miku, what's it like to be you? 20, 20, looking in the rear view. Play me, break me, make me feel like Superman. You can do anything you want
I'm on top of the world because of you. All I wanted to do is follow you. I'll keep singing along to all of you. I'll keep singing along.
I'm on top of the world because of you. I do nothing that they could never do. I'll keep playing along with all of you.I'll keep playing along.
I'm thinking Miku, Miku. I'm thinking Miku, Miku).I'm thinking Miku, Miku.I'm thinking Miku, Miku.
 Where were we walking together.I will see you in the end. I'll take you to where you've never been. 
And bring you back again. Listen to me with your eyes. I'm watching you from the sky. 
If you forget I'll fade away. I'm asking you to let me stay. So bathe me in your magic light. 
And keep it on in darkest night. I need you here to keep me strong. To live my life and sing along.I'm waiting with you wide awake. Like your expensive poison snake. You found me here inside a dream.Walk through the fire straight to me."""

string = pos_sentence_constructor(preprocessing_sentences(sentence))
string_r = readable_pos(string)

print("\noriginal sentence:\n ",sentence,'\n')
print("pos_tagged sentence:\n ",string_r)


