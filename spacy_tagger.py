import sys
import spacy
from nltk.tag import TaggerI
from nltk.corpus import treebank

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

data = treebank.tagged_sents()[3000:]

sp_tagger = SpacyTagger()
print(sp_tagger.evaluate(data))