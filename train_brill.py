import sys
from nltk.tag.brill import BrillTagger
from nltk.tag.brill_trainer import BrillTaggerTrainer
from nltk.tag import brill
from nltk.corpus import treebank
from nltk.tag import RegexpTagger

save_name = sys.argv[1]
tag = RegexpTagger([
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
    (r'(The|the|A|a|An|an)$', 'AT'),   # articles
    (r'.*able$', 'JJ'),                # adjectives
    (r'.*ness$', 'NN'),                # nouns formed from adjectives
    (r'.*ly$', 'RB'),                  # adverbs
    (r'.*s$', 'NNS'),                  # plural nouns
    (r'.*ing$', 'VBG'),                # gerunds
    (r'.*ed$', 'VBD'),                 # past tense verbs
    (r'.*', 'NN')                      # nouns (default)
])
templates = [
   brill.Template(brill.Pos([-1])),
   brill.Template(brill.Pos([1])),
   brill.Template(brill.Pos([-2])),
   brill.Template(brill.Pos([2])),
   brill.Template(brill.Pos([-2, -1])),
   brill.Template(brill.Pos([1, 2])),
   brill.Template(brill.Pos([-3, -2, -1])),
   brill.Template(brill.Pos([1, 2, 3])),
   brill.Template(brill.Pos([-1]), brill.Pos([1])),
   brill.Template(brill.Word([-1])),
   brill.Template(brill.Word([1])),
   brill.Template(brill.Word([-2])),
   brill.Template(brill.Word([2])),
   brill.Template(brill.Word([-2, -1])),
   brill.Template(brill.Word([1, 2])),
   brill.Template(brill.Word([-3, -2, -1])),
   brill.Template(brill.Word([1, 2, 3])),
   brill.Template(brill.Word([-1]), brill.Word([1])),
]
trainer = BrillTaggerTrainer(tag, templates)

train_data = treebank.tagged_sents()[:3000]
brill_tagger = trainer.train(train_data)

import pickle
pickle.dump(brill_tagger, open(save_name, "wb"))