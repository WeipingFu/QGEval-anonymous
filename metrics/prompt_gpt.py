import time
import numpy as np
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
import re

def completion(model, prompt, max_try=3, prt=False):
    client = OpenAI(
        base_url='',
        api_key=''
    )
    message = ''
    for i in range(max_try):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional evaluator!"},
                    {"role": "user", "content": prompt},
                ],
            )
            if prt:
                print('{} response:'.format(model))
                print(response)
            message = response.choices[0].message.content
            break
        except Exception as e:
            print(e)
            time.sleep(2)
    return message

def completion_n(model, prompt, max_try=3, prt=False):
    client = OpenAI(
        base_url='',
        api_key=''
    )
    all_messages = []
    for i in range(max_try):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional evaluator!"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=2,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=20,
                stop=None,
            )
            if prt:
                print('{} response:'.format(model))
                print(response)
            all_messages = [response.choices[i].message.content for i in
                                 range(len(response.choices))]
            break
        except Exception as e:
            print(e)
            time.sleep(2)
    return all_messages


class Criteria_GPT:
    def __init__(self) -> None:
        self.dimension_prompt = {
            'conciseness': 'Conciseness: Whether the question is concise and not abnormally verbose with redundant modifiers.\nEvaluation Criteria:\nscore 1: the question contains too much redundant information, making it difficult to understand its intent;\nscore 2: the question includes some redundant information, but it does not impact the understanding of its meaning;\nscore 3: the question is concise and does not contain any unnecessary information.',
            'fluency': 'Fluency: Whether the question is grammatically correct, coherent, and fluent enough to be understood.\nEvaluation Criteria:\nscore 1: the question is incoherent, with imprecise wording or significant grammatical errors, making it difficult to comprehend its meaning;\nscore 2: the question is slightly incoherent or contains minor grammatical errors, but it does not hinder the understanding of the question\'s meaning;\nscore 3: the question is fluent and grammatically correct.',
            'clarity': 'Clarity: Whether the question is expressed clearly and unambiguously, avoiding excessive generality and ambiguity.\nEvaluation Criteria:\nscore 1: the question is too broad or expressed in a confusing manner, making it difficult to understand or leading to ambiguity;\nscore 2: the question is not expressed very clearly and specifically, but it is possible to infer the question’s meaning based on the given passage;\nscore 3: the question is clear and specific, without any ambiguity.',
            'relevance': 'Relevance: Whether the question is relevant to the given passage and asks for key information from the passage.\nEvaluation Criteria:\nscore 1: the question is completely unrelated to the passage;\nscore 2: the question is somewhat related to the passage and it asks for non-crucial information related to the passage;\nscore 3: the question is relevant to the context, and the information it seeks is crucial to the passage.',
            'consistency': 'Consistency: Whether the information presented in the question is consistent with the passage and without any contradictions or hallucinations.\nEvaluation Criteria:\nscore 1: the question contains factual contradictions with the passage or logical errors;\nscore 2: the information sought in the question is not fully described in the passage;\nscore 3: the information in the question is entirely consistent with the passage.',
            'answerability': 'Answerability: Whether the question can be distinctly answered based on the passage.\nEvaluation Criteria:\nscore 1: the question cannot be answered based on the provided passage;\nscore 2: the question can be partially answered based on the provided passage or the answer to the question can be inferred to some extent;\nscore 3: the question can be answered definitively based on the given passage.',
            'answer consistency': 'Answer Consistency: Whether the question can be answered using the provided answer.\nEvaluation Criteria:\nscore 1: the question cannot be answered by the provided answer;\nscore 2: the question can be partially answered using the provided answer;\nscore 3: the question can be answered directly using the provided answer.'
        }
    
    def get_prompt(self, dimension, p, q, a, need_rational=False, prt=False):
        passage = 'Passage: ' + p
        question = 'Question: ' + q
        answer = 'Answer: ' + a
        criteria = self.dimension_prompt[dimension]
        if dimension in ['answer consistency']:
            task = 'You will be given a passage, a question, and a possible answer to this question. Your task is to evaluate whether the question meets the following requirement and score it on a scale from 1 to 3, with higher being better.'
            input_text = 'Now here are the passage, question, and answer:\n' + '\n'.join([passage, question, answer])
        else:
            task = 'You will be given a passage and a question. Your task is to evaluate whether the question meets the following requirement and score it on a scale from 1 to 3, with higher being better.'
            input_text = 'Now here are the passage and question:\n' + '\n'.join([passage, question])

        if need_rational:
            resp_format = 'Please response with format like "Score:your score; Rationale:your reason."'
        else:
            resp_format = 'Your response (score only):'

        prompt = '\n\n'.join([task, criteria, input_text, resp_format])
        if prt:
            print(prompt)
        return prompt

    def get_score(self, message, need_rational=False):
        score = -1
        message = message.lower()
        if need_rational:
            pattern = r'score:\s*(\d+)[,;]\s*rational:\s*(.+)'
        else:
            pattern = r'([1-3])+'
        match = re.search(pattern, message)
        if match:
            score = int(match.group(1))
        return score

    def get_score_float(self, message):
        score = -1
        message = message.lower()
        pattern = r'([1-3]\.?\d?)'
        match = re.search(pattern, message)
        if match:
            score = float(match.group(1))
        return score
    
    def get_scores(self, messages):
        score = -1
        scores = []
        pattern = r'([1-3]\.?\d?)'
        for message in messages:
            message = message.lower()
            match = re.search(pattern, message)
            if match:
                scores.append(float(match.group(1)))
        if len(scores) > 0:
            score = sum(scores) / len(scores)
        return score

    def gpt_one(self, model, dimension, p, q, a, need_rational=False, prt=False):
        one_res = {'message':[], 'score':-1}
        prompt = self.get_prompt(dimension, p, q, a, need_rational, prt)
        message = completion(model, prompt, prt=prt)
        # messages = completion_n(model, prompt, prt=prt)
        one_res['message'] = message
        one_res['score'] = self.get_score(message)
        # one_res['score'] = self.get_scores(messages)
        return one_res

    def gpt_batch(self, model, dimension, data, save_path=None, need_rational=False, prt=False):
        new_data = []
        new_col = dimension+'_'+model
        if need_rational:
            new_col = 'criteria-rational-'+model
        for idx, item in tqdm(enumerate(data), total=len(data)): 
            one_res = self.gpt_one(model, dimension, item['passage'], item['prediction'], item['answer'],
                                   need_rational=need_rational, prt=prt)
            item[new_col] = one_res
            new_data.append(item)
            if idx == 0:
                print('#'*10, 'Prompt', '#'*10)
                print(self.get_prompt(dimension,item['passage'],item['prediction'],item['answer'],need_rational))
                print('#'*10, 'Result', '#'*10)
                print(one_res)
            if idx % 1 == 0 and save_path is not None:
                pd.DataFrame(new_data).to_excel(save_path, index=False)
        if save_path is not None:
            pd.DataFrame(new_data).to_excel(save_path, index=False)
        return new_data


if __name__ == "__main__":
    cgpt = Criteria_GPT()
    model = 'gpt-4-1106-preview'
    # model = 'gpt-3.5-turbo'
    dimension = 'answerability'

    # one gpt
    p = '"3rd Bass was an American hip-hop group that rose to fame in the late 1980s and early 1990s, and was notable for being one of the first successful interracial hip-hop groups.  They split up in 1992 and again in 2000 after a failed reunion.  The group released two studio albums in their initial career and both of them were certified gold by the RIAA.\nThe Cactus Al/Bum (also known as The Cactus Cee/D and The Cactus Cas/Ette depending on release format) is the debut album by hip hop trio 3rd Bass, released on Def Jam Recordings on November 14, 1989.  The album received positive reviews from the hip hop press, and the group gained some publicity by being arguably the second white group to achieve hip hop credibility, after the Beastie Boys.  It was certified gold by the RIAA on April 24, 1990, the same day as Biz Markie\'s ""The Biz Never Sleeps"", which was released two weeks prior to ""The Cactus Album""."'
    q = 'How many albums did the group that released The Cactus Al/Bum release in their initial career?'
    a = 'two studio albums'
    # before apply, update base_url, and api_key in function completion
    print(cgpt.gpt_one(model, dimension, p, q, a, need_rational=False, prt=True))

    