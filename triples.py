import spacy

import coreferee
import  textacy
import pandas as pd
import json

from tools import read_clean, read_dictionary


def coreference_resolver(doc):

    # create list of tokens
    tok_list = [token.text for token in doc]
    
    # resolve corefs
    prev_shifts = 0
    for i, chain in enumerate(doc._.coref_chains):
    
        #print(f'\nchain : {doc._.coref_chains[i].pretty_representation}')
        for mention in chain:
        
            resolved1 = [doc._.coref_chains.resolve(doc[i]) for i in mention]
            resolved = list(filter((None).__ne__, resolved1))
            
            #print(f'mention : {mention} resolved : {resolved}')
            
            if len(resolved) > 0:
            
                # mention holds indices
                # resolved holds strings
                if len(resolved[0]) == 1:
                    tok_list[mention[0] + prev_shifts] = str(resolved[0][0])
                    
                elif len(resolved[0]) > 1:
                    tok_list[mention[0] + prev_shifts] = str(resolved[0][0])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                    
                    # loop through multi entity res
                    for i in range(1, len(resolved[0])):
                        tok_list.insert(mention[0] + i + prev_shifts , str(resolved[0][i]))
                        prev_shifts += 1
                
    return " ".join(tok_list)



# text input options
text_file = ['theColourOutOfSpace.txt',
            'election.txt',
            'election2.txt',
            ]

dict_file = '11172022_news.json'

samps = ['the fox ate dinner and breakfast.',
            'Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much.',
            'the fox jumped a fence.',
            'a fox painted his fence',
            'the fox and swan ate dinner',
            'the fox ate dinner and the swan ate lunch',
        ]



    # USE news dictionary
with open(dict_file) as infile:
    doc_dic = json.load(infile)

headlines = [val for val in list(doc_dic.keys())]
corpus = [value[1] for value in doc_dic.values()]

headline = headlines[0]
text = corpus[5]

    # USE txt file
#text = read_clean(text_file[2])

    # USE sample sentences
#text = samps[1]


# load spacy model
model = spacy.load("en_core_web_trf")
model.add_pipe('coreferee')


# create docs in both models
docs = list(model.pipe(corpus))


# coreference resolution on documents creates new text
# pass new text through model again
resolved_corpus = [coreference_resolver(doc) for doc in docs]
resolved_docs = list(model.pipe(resolved_corpus))

# extract SVO triples
#resolved_results = textacy.extract.triples.subject_verb_object_triples(resolved_docs)  
normal_results = textacy.extract.triples.subject_verb_object_triples(docs) 

'''
print(
        f'# of items in corpus : {len(corpus)}\n'
        f'# of items in norm_results : {len(normal_results)}\n'

    )
'''
#f'# of headlines : {len(list(headlines))}\n'
#f'# of items in resolved_results : {len(resolved_results)}\n' 

for i in range(len(headlines)):
    
    # headline title
    print(f'Headline : {headlines[i]}')
    
    # text
    print(f'Corpus : {corpus[i]}')
    
    # normal SVO results
    normal_results = textacy.extract.triples.subject_verb_object_triples(docs[i]) 
    for res in normal_results:
        print(res)
    
    print('\n')

    # resolved SVO results
    resolved_results = textacy.extract.triples.subject_verb_object_triples(resolved_docs[i])  
    for res in resolved_results:
        print(res)
    