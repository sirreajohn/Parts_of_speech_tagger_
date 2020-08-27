# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 23:45:27 2020

@author: mahesh
"""

def create_dictionaries(training_corpus  , vocab):
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    
    prev_tag = '--s--' 

    word_count = 0 
    for word_tag in training_corpus:

        word_count += 1
        if word_count % 50000 == 0:
            print(f"word count = {word_count}")
        word , tag = get_word_tag(word_tag ,vocab) 
        
        transition_counts[(prev_tag , tag)] += 1
        emission_counts[(tag , word)] += 1
        tag_counts[tag] += 1
        prev_tag = tag
    return emission_counts , transition_counts , tag_counts

def predict_pos(prep , emission_counts , vocab , states):
    pos_list  = []    
    all_words = set(emission_counts.keys())
    for word in prep: 
        count_final = 0
        pos_final = ''  
        if word in vocab:
            for pos in states:
                key = (pos ,word)
                if key in emission_counts:
                    count = emission_counts[key]
                    if count>count_final:
                        count_final = count
                        pos_final = pos
                        pos_list.append("<"+str(pos_final)+">")
    return pos_list

def pos_sentence_constructor(pos_token_list):
    string = ''
    for i in range(len(pos_token_list)):
        string +=str(pos_token_list[i])+' '
    return string

def readable_pos(string):
    re_string = ''
    for words in string.split():
        keys = get_key(words) 
        re_string += str(keys) + ' '
    return re_string

def get_key(val): 
    for key, value in dict_pos.items(): 
        for i in value:
         if val == i: 
             return key
    return '<UNK>'

def get_word_tag(line, vocabulary): 
    if not line.split():
        word = "--n--"
        tag = "--s--"
        return word, tag
    else:
        word, tag = line.split()
        if word not in vocabulary: 
            word = assign_unk(word)
        return word, tag
    return None 

def preprocess(vocabulary, data_fp):
    orig = []
    prep = []
    
    for cnt, word in enumerate(data_fp.split()):
        if not word:
            orig.append(word.strip())
            word = "--n--"
            prep.append(word)
            continue
        elif word.strip() not in vocabulary:
            orig.append(word.strip())
            word = assign_unk(word)
            prep.append(word)
            continue
        else:
            orig.append(word.strip())
            prep.append(word.strip())
    return orig, prep

def assign_unk(token):
    #handling unknown words
    if any(char.isdigit() for char in token):
        return "--unk_digit--"
    elif any(char in punct for char in token):
        return "--unk_punct--"
    elif any(char.isupper() for char in token):
        return "--unk_upper--"
    elif any(token.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"
    elif any(token.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"
    elif any(token.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"
    elif any(token.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"
    return "--unk--"

if __name__ == '__main__':
    from collections import defaultdict
    from nltk.corpus import brown
    import nltk
    import string
    punct = set(string.punctuation)
    # Morphology rules used to assign unknown word tokens
    noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
    verb_suffix = ["ate", "ify", "ise", "ize"]
    adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
    adv_suffix = ["ward", "wards", "wise"]    
    #training corpus tagging
    training_corpus = nltk.pos_tag(brown.words())
    training_corpus_list = []
    for i in training_corpus:
        training_corpus_list.append(str(i[0] + " " +i[1]))
        
    words = [line.split()[0] for line in training_corpus_list]
    freq = defaultdict(int)
    for word in words:
        freq[word] += 1
    vocabulary_dic = {}
    vocab = [k for k, v in freq.items() if (v > 1 and k != '\n')]
    for i , word in enumerate((sorted(vocab))):
        vocabulary_dic[word.lower()] = i 
    vocabulary_dic['Miku'] = 218748
    
    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus_list, vocabulary_dic)
    states = sorted(tag_counts.keys())
    
    sentence = """Miku , Miku , you can call me Miku. Blue hair , blue tie , hiding in your wi-fi  . Open secrets , anyone can find me  . Hear your music running through my mind
    I'm thinking Miku , Miku  . I'm thinking Miku , Miku  . I'm thinking Miku , Miku  .I'm thinking Miku , Miku  .
    I'm on top of the world because of you  . All I wanted to do is follow you  . I'll keep singing along to all of you  . I'll keep singing along
    I'm thinking Miku , Miku  .I'm thinking Miku , Miku)  .I'm thinking Miku , Miku  .I'm thinking Miku , Miku  .
    Miku , Miku , what's it like to be you? 20 , 20 , looking in the rear view  . Play me , break me , make me feel like Superman  . You can do anything you want
    I'm on top of the world because of you  . All I wanted to do is follow you  . I'll keep singing along to all of you  . I'll keep singing along  .
    I'm on top of the world because of you  . I do nothing that they could never do  . I'll keep playing along with all of you  .I'll keep playing along  .
    I'm thinking Miku , Miku  . I'm thinking Miku , Miku   .I'm thinking Miku , Miku  .I'm thinking Miku , Miku  .
    Where were we walking together  .I will see you in the end  . I'll take you to where you've never been  . 
    And bring you back again  . Listen to me with your eyes  . I'm watching you from the sky  . 
    If you forget I'll fade away  . I'm asking you to let me stay  . So bathe me in your magic light  . 
    And keep it on in darkest night  . I need you here to keep me strong  . To live my life and sing along  .I'm waiting with you wide awake  . Like your expensive poison snake  . You found me here inside a dream  .Walk through the fire straight to me  ."""
    
    _  , prep = preprocess(vocabulary_dic  , sentence)    
    
    pos_tagged = predict_pos(prep, emission_counts, vocabulary_dic, states)
    
    dict_pos = {'<conjuction>':['<CC>'],'<cardinal_digit>':['<CD>'],'<determiner>':['<DT>'],'<existential_there>':['<EX>'],'<foreign_word>':['<FW>'],
            '<preposition>':['<IN>'],'<adjective>':['<JJ>','<JJR>','<JJS>'],'<list_maker>':['<LS>'],'<modal>':['<MD>'],'<noun>':['<NN>','<NNS>','<NNP>','<NNPS>'],'<predeterminers>':['<PDT>','<POS>'],
            '<pronoun>':['<PRP>','PRP$'],'<adverb>':['<RB>','<RBR>','<RBS>','<RP>'],'<to>':['<TO>'],'<interjection>':['<UH>'],
            '<verb>':['<VB>','<VBD>','<VBN>','<VBP>','<VBZ>'],'<wh-word>':['<WDT>','<WP>','<WP$>','<WRB>'],'<punct>' :['<,>','<.>']}


    pos_tagged_sentence_str = pos_sentence_constructor(pos_tagged)
    pos_tagged_sentence = readable_pos(pos_tagged_sentence_str)
    
    