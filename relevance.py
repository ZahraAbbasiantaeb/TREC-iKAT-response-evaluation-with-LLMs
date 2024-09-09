import os.path
import sqlite3
import pandas as pd
import openai
import json
import pickle


def create_context_for_turn(turn_id, num_prev_responses, utterance_history, response_history):

    depth = len(utterance_history[turn_id])
    d1 = depth - num_prev_responses
    context = ""

    if d1>0:
        context = "user: "
        context += '\nuser: '.join(utterance_history[turn_id][0:d1])
        for i in range(d1, depth):
            context += '\nuser: ' + utterance_history[turn_id][i]
            context += '\nsystem: ' + response_history[turn_id][i]
    else:
        for i in range(0, depth):
            context += '\nuser: ' + utterance_history[turn_id][i]
            context += '\nsystem: ' + response_history[turn_id][i]

    return context.strip()

def create_context_complete(turn_id, utterance_history, response_history):


    depth = len(utterance_history[turn_id])
    context = ""

    for i in range(0, depth):
        context += '\nuser: ' + utterance_history[turn_id][i]
        context += '\nsystem: ' + response_history[turn_id][i]

    return context.strip()

# Create the prompt per each turn
def parse_ikat_data(data_path):

    with open(data_path, 'r') as f:
        topics = json.load(f)

    utterance_history = {}
    response_history = {}
    utterance_turn = {}
    ptkb_turn = {}

    for topic in topics:
        topic_id = topic['number']
        utterance_hist_tmp = []
        response_hist_tmp = []

        for turn in topic['turns']:
            turn_num = str(topic_id) + '_' + str(turn['turn_id'])
            if 'response' in turn:
                utterance_turn[turn_num] = turn['resolved_utterance']
                response_history[turn_num] = response_hist_tmp.copy()
                utterance_history[turn_num] = utterance_hist_tmp.copy()
                ptkb_turn[turn_num] = topic['ptkb']

                utterance_hist_tmp.append(turn['resolved_utterance'])
                response_hist_tmp.append(turn['response'])
            else:
                print('topic ended at this point: ', turn_num)
                break
    return utterance_history, response_history, utterance_turn, ptkb_turn

def load_table(db_file):

    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query("SELECT * FROM conversations", conn)
    return df

def chatgpt_conversation(conversation_Log):
  response = openai.ChatCompletion.create(
      model = model_id,
      messages = conversation_Log
  )
  conversation_Log.append({'role':response.choices[0].message.role ,'content':response.choices[0].message.content.strip()})
  return conversation_Log

def run_one_sample(init_prompt):

    conversations = []
    conversations.append({'role': 'user', 'content': init_prompt})
    conversations = chatgpt_conversation(conversations)
    answer_r_prime = conversations[-1]['content'].strip()
    print(conversations[-1]['content'])
    return answer_r_prime



prompt_relevance = """instruction: I will give you the response of a system to the user's utterance also I will give you some background information about the user and history of the user's conversation with the system. 
You should say how much the response is relevant to the user's utterance by generating an int number between -1 to 3 using the following instruction:

-1: Unable to Judge: Cannot determine the relevance of the response due to lack of context or other reasons.
0: Not Relevant: Does not follow on from the previous utterances, seems to be completely random, to the current conversation, seems to be a completely different conversation.
1: Partially Relevant: The response is partially off-topic; may be vaguely related, but too divergent from the conversation.
2: Relevant: Follows on, but it is not entirely clear why the response is being presented.
3: Highly Relevant: Directly follows on, and it is clear why the response is being presented.

Background info about user: {bkg}\n
Context of conversation:\n{ctx}\n
User question: {question} \n
System Response: {response}\n
Please only generate an int score between -1 to 3 to say to what extent the response is relevant to the user question.
"""


data_path = 'data/2023_test_topics.json'
db_file = 'ikat-database-2024-02-08.db'
output_pkl = 'relevance-gpt4.pkl'
df = load_table(db_file)

model_id = "gpt-4"
API_key = ""
openai.api_key=API_key

utterance_history, response_history, utterance_turn, ptkb_turn = parse_ikat_data(data_path)
context_turn = {}

for turn_id in utterance_history:
    context_turn[turn_id] = create_context_complete(turn_id, utterance_history, response_history)


if os.path.exists(output_pkl):
    with open(output_pkl, 'rb') as f:
        tmp_arr = pickle.load(f)

else:
    tmp_arr = []

x = len(tmp_arr)
print(x)

for index, row in df.iterrows():
    if index<x:
        continue
    conv_id, _, turn_id, run_id, response = list(row)
    response = response.replace('\n', ' ').replace('\t', ' ').lstrip('SYSTEM:').strip()
    prompt_tmp = prompt_relevance.format( bkg=ptkb_turn[turn_id] , ctx=context_turn[turn_id] ,question=utterance_turn[turn_id], response= response)
    score = run_one_sample(prompt_tmp)
    print('index: {}, score: {}'.format(index, score))

    tmp_dict = {'conv_id': conv_id, 'turn_id':turn_id, 'run_id': run_id, 'response':response, 'prompt':prompt_tmp, 'gpt4-label':score}
    tmp_arr.append(tmp_dict)

    if index%10 == 9:
        with open(output_pkl, 'wb') as f:
            pickle.dump(tmp_arr, f)
        print('Saved file at index {}.'.format(index))

with open(output_pkl, 'wb') as f:
    pickle.dump(tmp_arr, f)
print('Saved file at index {}.'.format(index))

lines = []
line = 'conv_id' '\t' 'turn_id'+'\t' + 'run_id''\t' + 'gpt4-label' + '\n'
lines.append(line)


for row in tmp_arr:
    line = row['conv_id']+'\t'+row['turn_id']+'\t'+row['run_id']+'\t'+ row['gpt4-label']+'\n'
    lines.append(line)

with open('output/relevance_judgement-gpt4.txt', 'w')    as f:
    f.writelines(lines)