# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from bs4 import BeautifulSoup
import json
from collections import Counter
import string
import argparse
import sys
import matplotlib.patches as mpatches
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
    
f=open("/Users/apple/Downloads/project_3/out_pred_coattention.txt","r",encoding='utf-8')
lines=f.readlines()
correct_count_dict={}
total_count_dict={}
correct_count_dict_f1={}

for line in lines: 
    span=line.split('\t')
    span[-1]=span[-1].strip()
    span_id=span[0]
    span_question=span[1]
    first_word=span_question.split()[0]
    span_num=span[2:]
    span_num=[int(x) for x in span_num]
    g_s=list(range(span_num[0],span_num[1]+1))
    p_s=list(range(span_num[2],span_num[3]+1))
    ground_truth_span=' '.join([str(x) for x in g_s])
    prediction_span=' '.join([str(x) for x in p_s])
    if first_word not in total_count_dict:
        total_count_dict[first_word]=1
    elif first_word in total_count_dict:
        total_count_dict[first_word]+=1
    if span_num[0]==span_num[2] and span_num[1]==span_num[3]:
        if first_word not in correct_count_dict:
                correct_count_dict[first_word]=1
                
        elif first_word in correct_count_dict:
                correct_count_dict[first_word]+=1
    if first_word not in correct_count_dict_f1:
        correct_count_dict_f1[first_word]=f1_score(prediction_span,ground_truth_span)
    else :
        correct_count_dict_f1[first_word]+=f1_score(prediction_span,ground_truth_span)
for x in total_count_dict.keys():
    if x not in correct_count_dict:
        correct_count_dict[x]=0
accuracy_rates={}
f1={}
for x in correct_count_dict.keys():
    accuracy_rates[x]=correct_count_dict[x]/total_count_dict[x]
    f1[x]=correct_count_dict_f1[x]/total_count_dict[x]
xx=[xx for xx in correct_count_dict.keys()]
yy=[accuracy_rates[x] for x in xx]
zz=[f1[x] for x in xx]
words = list(correct_count_dict.keys())
words[3]='How many'
y_pos = np.arange(len(words)) 
total_width, n = 0.8, 3
width = total_width / n
y_pos = y_pos - (total_width - width) / 2
plt.barh(y_pos, yy, width, label='EM')
plt.barh(y_pos+width, zz, width, label='F1')
plt.yticks(y_pos, words)
plt.legend()

plt.show()
