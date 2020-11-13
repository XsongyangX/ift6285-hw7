from nltk.tag import CRFTagger
from nltk.tag import brill
from nltk.tag.brill import brill24
from nltk.tag.brill_trainer import BrillTaggerTrainer
from nltk.corpus import treebank

# Taken from https://www.tutorialspoint.com/natural_language_toolkit/more_natural_language_toolkit_taggers.htm
# templates_tutorialspoint = [
#    brill.Template(brill.Pos([-1])),
#    brill.Template(brill.Pos([1])),
#    brill.Template(brill.Pos([-2])),
#    brill.Template(brill.Pos([2])),
#    brill.Template(brill.Pos([-2, -1])),
#    brill.Template(brill.Pos([1, 2])),
#    brill.Template(brill.Pos([-3, -2, -1])),
#    brill.Template(brill.Pos([1, 2, 3])),
#    brill.Template(brill.Pos([-1]), brill.Pos([1])),
#    brill.Template(brill.Word([-1])),
#    brill.Template(brill.Word([1])),
#    brill.Template(brill.Word([-2])),
#    brill.Template(brill.Word([2])),
#    brill.Template(brill.Word([-2, -1])),
#    brill.Template(brill.Word([1, 2])),
#    brill.Template(brill.Word([-3, -2, -1])),
#    brill.Template(brill.Word([1, 2, 3])),
#    brill.Template(brill.Word([-1]), brill.Word([1])),
# ]

crf = CRFTagger()
crf.set_model_file("crf.model")
templates = brill24()
trainer = BrillTaggerTrainer(crf, templates)

train_data = treebank.tagged_sents()[:3000]
brill_tagger = trainer.train(train_data)

test_data = treebank.tagged_sents()[3000:]
print(brill_tagger.evaluate(test_data))