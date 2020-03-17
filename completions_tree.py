# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:16:13 2020
@author: Doug Summers Stay
Find the most probable completions of a sentence using gpt-2.
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import re

def grow_branches(sentence_so_far, probs, input_probability,past, h):
    #recursive function to find all sentence completions
    global branch_list
    global leaf_list
    global complete_list
    global model
    sorted_probability_list = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    has_children = False
    for (this_token,this_probability) in sorted_probability_list:
        next_probability = this_probability * input_probability
        out_sentence = sentence_so_far.copy()
        sentence_and_probability = (out_sentence, input_probability)
        pattern = ' [A-Z]{1,1}'
        pattern2 = '[A-Z]{1,1}'
        test_string = tokenizer.decode(out_sentence[-1])
        result = re.match(pattern, test_string) or re.match(pattern2, test_string)   
        if not (result or (out_sentence[-1] in {1583,1770,6997,19090,9074,7504})) and (this_token == 13):
            #if the next token is going to be a period, then no need to carry out that step.
            #except allow Mr., Dr., Mrs., Ms., Lt., Sgt., Jr. or single initials.
            sentence_and_probability = (out_sentence, next_probability)
            complete_list.append(sentence_and_probability)
            return
        if next_probability < h:            
            if has_children == True:
                branch_list.append(sentence_and_probability)
            else:
                leaf_list.append(sentence_and_probability)
            return
        else:
            has_children = True
            next_sentence = sentence_so_far.copy()
            next_sentence.append(this_token)
            (next_probability_list,next_past) = expand_node(next_sentence,past)
            grow_branches(next_sentence,next_probability_list, next_probability, next_past, h)

def expand_node(sentence, past):
    #finds probabilities for the next token using gpt-2
    global model
    if past == None:
        input_ids = torch.tensor(sentence).unsqueeze(0)
    else:
        input_ids = torch.tensor([sentence[-1]]).unsqueeze(0)
    inputs = {'input_ids': input_ids}    
    with torch.no_grad():
        logits, past = model(**inputs, past=past)   
        logits = logits[:, -1, :]  
        probs = F.softmax(logits, dim=-1).tolist()[0]
        return (probs, past)

# globals here
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


leaf_list = []
branch_list = []
complete_list = []
probability_threshhold=float(input("probability cutoff (e.g. .001 or less):"))
raw_prompt = input("partial sentence to complete:")

prompt=tokenizer.encode(raw_prompt)

(probs, past) = expand_node(prompt, None) 
grow_branches(prompt,probs,1,past,probability_threshhold)

sorted_complete_list = sorted(complete_list, reverse=True,key=lambda x: x[1])
sorted_leaf_list = sorted(leaf_list, reverse=True,key=lambda x: x[1])
sorted_branch_list = sorted(branch_list, reverse=True,key=lambda x: x[1])

# to get the most probable completed sentence:
# tokenizer.decode(sorted_complete_list[0])

#print just the completions
for (sentence, prob) in sorted_complete_list:
    #print(round(prob,6),end=':')
    if prob>probability_threshhold:
        print(repr(tokenizer.decode(sentence[len(prompt):])).strip("'"),end='|')
    else:
         print(repr(tokenizer.decode(sentence[len(prompt):])).strip("'"),end='\\')


