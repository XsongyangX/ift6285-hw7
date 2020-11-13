from nltk.tag import CRFTagger
from nltk.tag import brill
from nltk.tag.brill_trainer import BrillTaggerTrainer
from nltk.corpus import treebank

crf = CRFTagger()
crf.set_model_file("crf.model")
trainer = BrillTaggerTrainer(crf, [brill.Template(brill.Pos([-1]))])

train_data = treebank.tagged_sents()[:3000]
brill_tagger = trainer.train(train_data)

test_data = treebank.tagged_sents()[3000:]
print(brill_tagger.evaluate(test_data))