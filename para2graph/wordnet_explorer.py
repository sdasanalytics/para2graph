#----------------------------#
# Author: Surjit Das
# Email: surjitdas@gmail.com
# Program: artmind
#----------------------------#

from nltk.corpus import wordnet as wn
import nltk

def download_wordnet_corpora():
    nltk.download("wordnet")
    nltk.download("omw-1.4")

class WordNet_Explorer():
    def __init__(self, text) -> None:
        self.synsets = wn.synsets(text)

    def print_info(self):
        for synset in self.synsets:
            print(f"{synset.name()=}")
            print(f"{synset.definition()=}")
            print(f"{synset.examples()=}")
            print(f"{synset.lemma_names('eng')=}")
            hypernyms = [hypernym.name() for hypernym in synset.hypernyms()]
            print(f"{hypernyms=}")
            print("----------")

def test(text):
    wn_exp = WordNet_Explorer(text)
    wn_exp.print_info()

if __name__=="__main__":
    # download_wordnet_corpora() # Will need to execute this for the first time (if corpora not downloaded)
    test("song")
    