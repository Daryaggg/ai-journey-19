# Code written by my teammate Aleksandr Zhelubenkov

import json
import os
from glob import glob

import numpy as np
import requests


def get_correct(task):
    if "correct_variants" in task["solution"]:
        if isinstance(task["solution"]["correct_variants"][0], str):
            correct = set(task["solution"]["correct_variants"])
        else:
            correct = set(task["solution"]["correct_variants"][0])
    elif "correct" in task["solution"]:
        if isinstance(task["solution"]["correct"], str):
            correct = {task["solution"]["correct"]}
        else:
            correct = set(task["solution"]["correct"])
    else:
        raise ValueError("Unknown task format!")
    return correct


def check_solution(task, solution, task_id=None):
    if task_id is None:
        task_id = task.get('id')

    task_type = task["question"]["type"]
    if task_type == "matching":
        y_true = task['solution']['correct']
        score = get_matching_score(y_true, solution)
        return score, y_true
    elif task_id == '16':
        y_true = (
            task["solution"]["correct_variants"][0]
            if "correct_variants" in task["solution"]
            else task["solution"]["correct"]
        )
        score = get_multiple_score(y_true, solution)
        return score, y_true
    else:
        correct = get_correct(task)
        if task['question']['type'] == 'text':
            return int(solution in correct), correct
        else:
            return int(set(solution) == correct), correct


def get_matching_score(y_true, pred):
    score = 0
    if len(y_true) != len(pred):
        return 0
    for key in y_true.keys():
        if y_true[key] == pred.get(key):
            score += 1
    return score


def get_multiple_score(y_true, pred):
    score = 0
    for y in y_true:
        for p in pred:
            if y == p:
                score += 1
    return score


class Checker:
    def __init__(self, data_dirname, is_nested=True):
        self.data = dict()

        if is_nested:
            for dirname in glob(os.path.join(data_dirname, '*')):
                for file in glob(os.path.join(dirname, '*')):
                    with open(file, encoding='utf-8') as fin:
                        k = os.path.split(file)[-1].replace('.json', '')
                        self.data[k] = json.load(fin)
                        if 'tasks' not in self.data[k]:
                            self.data[k] = {'tasks': [self.data[k]]}
        else:
            for file in glob(os.path.join(data_dirname, '*')):
                with open(file, encoding='utf-8') as fin:
                    k = os.path.split(file)[-1].replace('.json', '')
                    self.data[k] = json.load(fin)
                    if 'tasks' not in self.data[k]:
                        self.data[k] = {'tasks': self.data[k]}

    def get_tasks(self, task_id=None):
        for e in self.data:
            for task in self.data[e]['tasks']:
                if task['id'] == str(task_id) or task_id is None:
                    yield task

    def check(self, task_id, model):
        return self._check(task_id, model, 'solve')

    def check_sber_baseline(self, task_id, model):
        return self._check(task_id, model, 'sber')

    def check_flask(self, task_id):
        return self._check(task_id, None, 'flask')

    def _check(self, task_id, model, mode):
        scores = []
        info = []
        for task in self.get_tasks(task_id):
            if mode == 'solve':
                res = model.solve(task, task_id)
            elif mode == 'sber':
                res = model.take_exam([task])[str(task_id)]
            elif mode == 'flask':
                resp = requests.post('http://localhost:8000/take_exam', json=[task])
                js = json.loads(resp.content)['answers']
                res = js[str(task_id)]
            else:
                raise

            score, correct = check_solution(task, res)
            scores.append(score)
            info.append({'task': task, 'correct': correct, 'res': res, 'score': scores[-1]})

        return np.mean(scores), len(scores), scores, info
