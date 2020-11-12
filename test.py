import sys

from nltk.corpus import treebank
from nltk.tag import CRFTagger

save_name = sys.argv[1]

ct = CRFTagger()
# train_data = treebank.tagged_sents()[:3000]
test_data = treebank.tagged_sents()[3000:]

# test
ct.set_model_file(save_name)
print(ct.evaluate(test_data))
