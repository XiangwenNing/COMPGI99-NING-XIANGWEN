from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
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

f=open("/Users/apple/Downloads/project_3/out_pred6.txt","r",encoding='utf-8')
lines=f.readlines()
correct_count=0
total_count=0
f1=0
for line in lines: 
    span=line.split('\t')
    span[-1]=span[-1].strip()
    span_id=span[0]
    span_num=span[2:]
    g_s=list(range(int(span_num[0]),int(span_num[1])+1))
    p_s=list(range(int(span_num[2]),int(span_num[3])+1))
    ground_truth_span=' '.join([str(x) for x in g_s])
    prediction_span=' '.join([str(x) for x in p_s])
    f1=f1+f1_score(prediction_span,ground_truth_span)
    total_count+=1
    if span_num[0]==span_num[2] and span_num[1]==span_num[3]:
        correct_count+=1
ave_f1=f1/total_count
print(ave_f1)
ave_acc=correct_count/total_count
print(ave_acc)

