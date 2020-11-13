import sys
from nltk.tag.brill import BrillTagger, brill, brill24
import spacy
from nltk.tag import TaggerI
from nltk.corpus import treebank
from nltk.tag.brill_trainer import BrillTaggerTrainer
import time
start_time = time.time()

class SpacyTagger(TaggerI):
    
    def __init__(self):
        """
        Creates a spacy instance
        """
        model = "en_core_web_sm" # try also the _lg one
        self.nlp = spacy.load(model,
            disable=["parser", "ner"]) # to go faster

    def tag(self, tokens):
        """
        Overrides tag from the interface
        """
        # we want to do this:
        # doc = nlp('hello world !')
        #
        # but the tokenization would change from the one in treebank
        # which would cause problems with the function evaluate
        # so instead do this more convoluted thing:
        doc = spacy.tokens.doc.Doc(self.nlp.vocab, words=tokens)
        for _, proc in self.nlp.pipeline:
            doc = proc(doc)
        # now doc is ready:
        return [(t.text, t.tag_) for t in doc]

# save_name = sys.argv[1]

train_data = treebank.tagged_sents()[:3000]
test_data = treebank.tagged_sents()[3000:]

recursive_tagger : BrillTagger = None

# Clean up templates
brill.Template._cleartemplates()
templates = brill24()

for i in range(sys.argv[1]):
  if i == 0:
    trainer = BrillTaggerTrainer(SpacyTagger(), templates)
  else:
    trainer = BrillTaggerTrainer(recursive_tagger, templates)
  recursive_tagger = trainer.train(train_data)
  print(f"Iteration {i+1}, time elapsed: {time.time() - start_time}")
  print(f"Train score: {recursive_tagger.evaluate(train_data)}")
  print(f"Test score: {recursive_tagger.evaluate(test_data)}")