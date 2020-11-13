import sys

from nltk.corpus import treebank

save_name = sys.argv[1]

# train_data = treebank.tagged_sents()[:3000]
test_data = treebank.tagged_sents()[3000:]

# test
from nltk.tag.brill import BrillTagger
from pickle import load

encoded = load(open(save_name, "rb"))
brill_tagger = BrillTagger.decode_json_obj(encoded)

print(brill_tagger.evaluate(test_data))
