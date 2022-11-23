import spacy
import coreferee
import  textacy
import json
import cupy as cp
import time

from tools import read_clean, read_dictionary


# 274.6 sec w GPU
# 
# Start time
start = time.perf_counter()


# Coreferee custom attirbutes allow for coreference resolution. 
# Input : spacy doc type
# Output : new text string that has been resolved
def coreference_resolver(doc):

    tok_list = [token.text for token in doc]
    
    # resolve corefs   
    prev_shifts = 0
    for i, chain in enumerate(doc._.coref_chains):
    
        # mention holds indices
        # resolved holds strings        
        for mention in chain:
        
            resolved1 = [doc._.coref_chains.resolve(doc[i]) for i in mention]
            resolved = list(filter((None).__ne__, resolved1))
            
            if len(resolved) > 0:
            
                if len(resolved[0]) == 1:
                    tok_list[mention[0] + prev_shifts] = str(resolved[0][0])
                    
                elif len(resolved[0]) > 1:
                    tok_list[mention[0] + prev_shifts] = str(resolved[0][0])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                    
                    # loop through multi entity res
                    for i in range(1, len(resolved[0])):
                        tok_list.insert(mention[0] + i + prev_shifts , str(resolved[0][i]))
                        prev_shifts += 1
                
    return " ".join(tok_list)


# Simplify complex sentences
# Input : spacy doc type
# Output : simple sentence list
def compound_to_simple(doc):

    for token in doc:
        if (token.dep_ == "ROOT"):
            root_token = token
            
    other_verbs = []
    for token in doc:
        ancestors = list(token.ancestors)
        if (token.pos_ == "VERB" and len(ancestors) < 3 and token != root_token):
            other_verbs.append(token)
            
    token_spans = []
    all_verbs = [root_token] + other_verbs
    for other_verb in all_verbs:
        first_token_index = len(doc)
        last_token_index = 0
        this_verb_children = list(other_verb.children)
        for child in this_verb_children:
            if (child not in all_verbs):
                if (child.i < first_token_index):
                    first_token_index = child.i
                if (child.i > last_token_index):
                    last_token_index = child.i
        token_spans.append((first_token_index, last_token_index))
        
    sentence_clauses = []
    for token_span in token_spans:
        start = token_span[0]
        end = token_span[1]
        if (start < end):
            clause = doc[start:end]
            sentence_clauses.append(clause)
    sentence_clauses = sorted(sentence_clauses, key=lambda tup: tup[0])
    clauses_text = [clause.text for clause in sentence_clauses]
    
    return clauses_text


spacy.require_gpu()

INPUT_DIRECTORY = 'texts\\'
OUTPUT_DIRECTORY = 'results\\'

# text input options
text_file = ['theColourOutOfSpace.txt',
            'election.txt',
            'election2.txt',
            ]

# file input / output details
dict_file = '112122_news.json'
dict_name = dict_file.split('.')[0]
result_type = 'result_SVO'


samps = ['the fox ate dinner and breakfast.',
            'Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much.',
            'the fox jumped a fence.',
            'a fox painted his fence',
            'the fox and swan ate dinner',
            'the fox ate dinner and the swan ate lunch',
        ]

# Use news dictionary
with open(INPUT_DIRECTORY + dict_file, encoding='utf-8') as infile:
    doc_dic = json.load(infile)

headlines = [val for val in list(doc_dic.keys())]
corpus = [value[1] for value in doc_dic.values()]

# USE .txt file
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


result_str = ''
for i in range(len(headlines)):
    
    result_str += f'\nHeadline : {headlines[i]}\n\n'
    result_str += f'Corpus : {corpus[i]}\n\n'
    
    # simplify coref resolved results
    simplified_texts = compound_to_simple(resolved_docs[i])
    simplified_docs = list(model.pipe(simplified_texts))


    # SVO on simplified clauses
    # note: SVO extraction requires doc
    #resolved_results = textacy.extract.triples.subject_verb_object_triples(simplified_docs)
    resolved_results = [textacy.extract.triples.subject_verb_object_triples(doc) for doc in simplified_docs]  
    
    for clause in resolved_results:
        for res in clause:
            result_str += f'{str(res)}\n'

    
    result_str += '\nno coref, no simplified SVO results\n\n'
    # normal SVO results
    normal_results = textacy.extract.triples.subject_verb_object_triples(docs[i]) 
    for res in normal_results:
        result_str += f'{str(res)}\n'
    
    result_str += '\n\n'


# Computation finish time
comp_finish  = time.perf_counter()
print(f'Elapsed time : {comp_finish-start:0.2f}')

with open(OUTPUT_DIRECTORY + dict_name + result_type + '.txt', 'w', encoding='utf-8') as f:
    f.write(result_str)

# Finish time
finish  = time.perf_counter()
print(f'Elapsed time : {finish-start:0.2f}')