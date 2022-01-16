#----------------------------#
# Author: Surjit Das
# Email: surjitdas@gmail.com
# Program: artmind
#----------------------------#

# References:
# - https://medium.com/broken-window/the-power-of-wordnet-with-nltk-7c45b20f52cf
# - https://www.nltk.org/howto/wordnet.html

from nltk.corpus import wordnet as wn
import nltk
import sys
from loguru import logger as log

def download_wordnet_corpora():
    # By default these should get downloaded into /Users/surjitdas/nltk_data
    nltk.download("wordnet")
    nltk.download("omw-1.4")

class WordNet_Explorer():
    def __init__(self, text) -> None:
        self.synsets = wn.synsets(text)

    def get_parent_classes(self):
        parents = []
        if len(self.synsets) == 0:
            return parents
        
        synset = self.synsets[0]
        while True:
            try:
                synset = synset.hypernyms()[-1]
                parents.append(synset.name().split('.')[0])
            except IndexError:
                break
        if len(parents) == 0:
            try:
                lemma1 = synset.lemmas()[0]
                # print(f"{lemma1=}")
                related_form1 = lemma1.derivationally_related_forms()[0].name()
                # print(f"{related_form1=}")
                related_synset = wn.synsets(related_form1)[0]
                while True:
                    try:
                        related_synset = related_synset.hypernyms()[-1]
                        parents.append(related_synset.name().split('.')[0])
                    except IndexError:
                        break                
            except:
                log.debug("Error while getting parent classes")
        return parents

    def print_info(self):
        for synset in self.synsets:

            print(f"{synset.name()=}")
            print(f"{synset.definition()=}")
            print(f"{synset.examples()=}")
            hypernyms = [hypernym.name() for hypernym in synset.hypernyms()]
            print(f"{hypernyms=}")
            # print(f"{synset.lemma_names('eng')=}")
            for lemma in synset.lemmas():
                print(f"{lemma.name()=}")
                for related_form in lemma.derivationally_related_forms():
                    print(f"\t{related_form=}")
                for pertainym in lemma.pertainyms():
                    print(f"\t{pertainym=}")
                for antonym in lemma.antonyms():
                    print(f"\t{antonym=}")
            print("----------")

    def similarity_with(self, another):
        first_synset = self.synsets[0]
        second_synset = wn.synsets(another)[0]
        path_similarity = first_synset.path_similarity(second_synset)
        lch_similarity = first_synset.lch_similarity(second_synset)
        print(f"Between {first_synset} & {second_synset} the similarity scores are: {path_similarity=}, {lch_similarity=}")

        
def test(text1="cat", text2="entity"):
    wn_exp = WordNet_Explorer(text1)
    parents = wn_exp.get_parent_classes()
    print(f"{parents=}")
    wn_exp.print_info()
    wn_exp.similarity_with(text2)


if __name__=="__main__":
    # download_wordnet_corpora() # Will need to execute this for the first time (if corpora not downloaded)
    # test("cat", "dog")
    if len(sys.argv) ==2:
        test(sys.argv[1])
    else:
        test()
    