import sys
from typing import List

from nltk.corpus import treebank
from nltk.tag import CRFTagger

save_name = sys.argv[1]

# custom feature function
# observes a window of tokens
dummy_tagger = CRFTagger()
def window(tokens: List[str], idx:int) -> List[str]:
    token_of_interest = dummy_tagger._feature_func(tokens, idx)

    # get previous token
    if idx < len(tokens) - 1:
        previous_token = dummy_tagger._feature_func(tokens, idx+1)
        token_of_interest.extend([f"NEXT_{feature}" for feature in previous_token])

    return token_of_interest

ct = CRFTagger(feature_func=window)


train_data = treebank.tagged_sents()[:3000]
ct.train(train_data, save_name)
