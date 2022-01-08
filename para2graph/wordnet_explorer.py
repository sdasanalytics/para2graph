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

def download_wordnet_corpora():
    nltk.download("wordnet")
    nltk.download("omw-1.4")

class WordNet_Explorer():
    def __init__(self, text) -> None:
        self.synsets = wn.synsets(text)

    def get_parent_classes(self):
        synset = self.synsets[0]
        print(f"Parent classes of {synset.name()}:")
        while True:
            try:
                synset = synset.hypernyms()[-1]
                print(synset)
            except IndexError:
                print("----------")
                break 

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

        
def test(text1, text2="entity"):
    wn_exp = WordNet_Explorer(text1)
    wn_exp.get_parent_classes()
    wn_exp.print_info()
    wn_exp.similarity_with(text2)


if __name__=="__main__":
    # download_wordnet_corpora() # Will need to execute this for the first time (if corpora not downloaded)
    # test("cat", "dog")
    test("cat")
    