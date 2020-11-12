import sys

from nltk.corpus import treebank

save_name = sys.argv[1]

# train_data = treebank.tagged_sents()[:3000]
test_data = treebank.tagged_sents()[3000:]

# test
from nltk.tag import CRFTagger
ct = CRFTagger()

ct.set_model_file(save_name)
print(ct.evaluate(test_data))
