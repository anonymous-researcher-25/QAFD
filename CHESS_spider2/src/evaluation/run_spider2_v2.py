from evaluation_v2 import ExecutionAccuracy
import pandas as pd
import json
import random

path = open('', 'r')
all_questions = json.load(path)['sqlite'] # list of dicts with tasks
csv_path = ''

test = {}
infer = {}
dbs = {}
# import pdb 
# pdb.set_trace()
# breakpoint()
# for q in all_questions: # q is a task
#     sql = all_questions[q][0]
#     if sql:
#         infer[q] = sql
#         test[q] = f'{csv_path}{all_questions[q][1]}.csv'
#         dbs[q] = all_questions[q][2]
#
# Change
#######################################
for one in all_questions:
    q = one['question']
    infer[q] = one['sql']
    dbs[q] = one['db_name']
    test[q] = one['csv']
#######################################

args = {
    'db_type': 'sqlite3',
    'host': '0.0.0.0',
    'userid': 'root',
    'pwd': 'root',
    'data_name': 'spider2',
    'db_path': '',
    'gold_type': 'csv'
}
exesql = ExecutionAccuracy(args)
results = exesql.run(infer, test, dbs)

correct, wrong = [], []
for q in results:
    if results[q]['exec_err'] == '--':
        correct.append(q)
    else:
        wrong.append(q)
        
print(len(correct), len(wrong))
print(wrong)