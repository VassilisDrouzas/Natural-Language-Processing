# -*- coding: utf-8 -*-

import pandas as pd
import conllu
from tqdm.auto import tqdm

import itertools


def conllu_to_pd(file_path: str) -> pd.DataFrame:
    """
    Convert a CoNLL-U file to a Pandas DataFrame.
    :param file_path: Path to the CoNLL-U file.
    :return: A DataFrame with columns 'words', 'pos', and 'sent_id'.
    """
    print("\tReading data...")
    with open(file_path, "r") as file:
        data = file.read()

    print("\tParsing data...")
    conllu_sentences = conllu.parse(data)

    print("\tGetting words...")
    words = [[word["form"].lower() for word in sentence] 
             for sentence in tqdm(conllu_sentences)]

    print("\tGetting POS tags...")
    pos = [[word["upos"] for word in sentence] 
           for sentence in tqdm(conllu_sentences)]

    print("\tGetting Sentence ids...")
    sent_ids = [[sent.metadata["sent_id"]] * len(sent) 
                for sent in tqdm(conllu_sentences)]

    return pd.DataFrame({"words": itertools.chain.from_iterable(words),
                         "pos": itertools.chain.from_iterable(pos),
                         "sent_id": itertools.chain.from_iterable(sent_ids)})
