import re
from typing import Any, Dict, List, Tuple

from code_snippets.utils import MyMorph
from sberbank_baseline.data_processing import NgramManager

WordWindow = Tuple[List[str], str, List[str]]


def get_sentences_from_text(file_name: str, min_symbols_in_sent: int = 10) -> List[str]:
    with open(file_name, encoding='UTF8') as file:
        text = file.read()

    text = text.split('* * *')[2].lower()
    sentences = re.split('\.|\?|!|\\n', text)
    sentences = list(filter(lambda x: len(x) >= min_symbols_in_sent, sentences))
    return sentences


def sentance_to_windows(sentence: str, n_backw: int, n_frontw: int, min_words_in_sent: int = 5) -> List[WordWindow]:
    sentence = sentence.split(" ")
    n = len(sentence)
    if n < min_words_in_sent:
        return []

    windows = []
    for i in range(n - 1):
        windows.append(
            (
                sentence[max(0, i - n_backw) : i],
                sentence[i],
                sentence[i + 1 : min(n, i + n_frontw + 1)],
            )
        )
    return windows


def vectorizer(
    word_window: WordWindow,
    morph: MyMorph,
    ngram_man: NgramManager,
    save_word_pos_list: List[str] = ['PREP', 'CONJ', 'PRCL', 'INTJ']
) -> Dict[str, Any]:
    words_backw, main_word, words_front = word_window
    feature_dict = dict()

    # parse commas after main word in window to get taget value for learning
    feature_dict['y'] = 1 if main_word.endswith(',') else 0

    # get features from main word
    main_word = main_word.strip(',')
    tag = morph.get_tag(main_word)
    feature_dict['main_tag'] = tag.POS
    # if any tag of word classifies it as Function word, save it as additional feature
    if any([_ in tag for _ in save_word_pos_list]):
        feature_dict['main_word'] = main_word

    feature_dict['main_case'] = tag.case
    feature_dict['main_aspect'] = tag.aspect

    # get features from back words, if they exist
    if words_backw:
        for i, word in enumerate(words_backw[::-1]):
            tag = morph.get_tag(word.strip(','))
            feature_dict[f'b_{i+1}_tag'] = tag.POS
            feature_dict[f'b_{i+1}_case'] = tag.case
            feature_dict[f'b_{i+1}_aspect'] = tag.aspect

    # get features from front words, if they exist
    if words_front:
        # save firts word after comma as additional feature, too, if it relates to Function words
        tag = morph.get_tag(words_front[0].strip(','))
        if any([_ in tag for _ in save_word_pos_list]):
            feature_dict['f_1_word'] = words_front[0].strip(',')

        for i, word in enumerate(words_front):
            tag = morph.get_tag(word.strip(','))
            feature_dict[f'f_{i+1}_tag'] = tag.POS
            feature_dict[f'b_{i+1}_case'] = tag.case
            feature_dict[f'b_{i+1}_aspect'] = tag.aspect

    # add ngram frequencies as features: find out how often ngram occurs with and wo comma
    if words_front:
        # 2gram: main word + next word
        feature_dict['2_ngram_1'] = ngram_man.get_freq(
            tuple([main_word.strip(','), '', words_front[0].strip(',')])
        )
        feature_dict['2_ngram_0'] = ngram_man.get_freq(
            tuple([main_word.strip(','), ',', words_front[0].strip(',')])
        )

        # 3gram: previous word + main word + next word
        if words_backw:
            feature_dict['31_ngram_0'] = ngram_man.get_freq(
                tuple(
                    [
                        words_backw[-1].strip(','),
                        '',
                        main_word.strip(','),
                        '',
                        words_front[0].strip(','),
                    ]
                ),
            )
            feature_dict['31_ngram_1'] = ngram_man.get_freq(
                tuple(
                    [
                        words_backw[-1].strip(','),
                        '',
                        main_word.strip(','),
                        ',',
                        words_front[0].strip(','),
                    ]
                ),
            )

        # 3gram: main word + 2 next words
        if len(words_front) > 1:
            feature_dict['32_ngram_0'] = ngram_man.get_freq(
                tuple(
                    [
                        main_word.strip(','),
                        '',
                        words_front[0].strip(','),
                        '',
                        words_front[1].strip(','),
                    ]
                ),
            )
            feature_dict['32_ngram_1'] = ngram_man.get_freq(
                tuple(
                    [
                        main_word.strip(','),
                        ',',
                        words_front[0].strip(','),
                        '',
                        words_front[1].strip(','),
                    ]
                ),
            )

    return feature_dict
