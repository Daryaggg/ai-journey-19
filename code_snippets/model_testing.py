import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from code_snippets.checker import Checker
from code_snippets.data_processing import vectorizer
from code_snippets.utils import MyMorph
from sberbank_baseline.data_processing import NgramManager


def word_adder(list_words: List[str], n_words: int = 5) -> List[str]:
    final_words = []
    i = 0
    while (len(final_words) < n_words) and (i < len(list_words)):
        if len(re.findall('([1-9])', list_words[i])) == 0:
            final_words.append(list_words[i])
        i += 1
    return final_words


def test_model(
    model: Any,
    checker: Checker,
    morph: MyMorph,
    ngram_man: NgramManager,
    thresholds: List[float],
    X_df_columns: List[str],
    tasks: List[int] = [17, 18, 19, 20],
    n_words: int = 5,
) -> Dict[str, List[int]]:
    t_answ = dict()
    for t in thresholds:
        print('Predictions for threshold:', t)
        print()

        answ = []
        for task in tasks:
            for ex in checker.get_tasks(task):
                task_text = re.split('\n|\xa0', ex['text'])
                sentences = [s for s in task_text if len(re.findall('([1-9])', s)) > 0]

                word_windows = []
                for sentence in sentences:
                    sentence = sentence.split(" ")
                    n = len(sentence)

                    windows = []
                    for i in range(n):
                        if len(re.findall('([1-9])', sentence[i])) > 0:
                            windows.append(
                                (
                                    word_adder(sentence[max(0, i - n_words * 2) : i - 1]),
                                    sentence[i - 1],
                                    word_adder(sentence[i + 1 : min(n, i + n_words * 2)]),
                                )
                            )
                    word_windows += windows
                X_df_checker = pd.DataFrame([vectorizer(w, morph, ngram_man) for w in word_windows])
                X_df_checker = pd.get_dummies(X_df_checker).fillna(0)
                cols = set(X_df_columns) - set(X_df_checker.columns)
                for col in cols:
                    X_df_checker[col] = 0

                X_df_checker = X_df_checker[X_df_columns]
                y_prob = model.predict(X_df_checker)
                y_checker = [1 if _ > t else 0 for _ in y_prob]

                print(ex['text'])

                # Mostly sentences contain minimum 1 comma
                # so if model predicts no commas, we'll set it in a random position
                if all([yi == 0 for yi in y_checker]):
                    print('Bad prediction: No commas')
                    i = np.random.choice(range(len(y_checker)))
                    y_checker[i] = 1

                print({'prediction': [i + 1 for i, yi in enumerate(y_checker) if yi == 1]})
                print(ex['solution'])
                print()

                if 'correct_variants' in ex['solution'].keys():
                    if str([str(i + 1) for i, yi in enumerate(y_checker) if yi == 1]) in [
                        str(_) for _ in ex['solution']['correct_variants']
                    ]:
                        answ.append(1)
                    else:
                        answ.append(0)
                elif 'correct' in ex['solution'].keys():
                    if str([str(i + 1) for i, yi in enumerate(y_checker) if yi == 1]) == str(
                        ex['solution']['correct']
                    ):
                        answ.append(1)
                    else:
                        answ.append(0)
        print(t, np.mean(answ))
        print()

        t_answ[t] = answ

    return t_answ
