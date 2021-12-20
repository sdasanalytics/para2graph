#----------------------------#
# Author: Surjit Das
# Email: surjitdas@gmail.com
# Program: artmind
#----------------------------#

from dataclasses import dataclass
import constants as C
import py2neo as p2n
from loguru import logger as log
import py2neo as p2n

class SentenceGraph:
    def __init__(self, G_n4j, sentence_uuid) -> None:
        self.G_n4j = G_n4j
        self.sentence_uuid = sentence_uuid

    def save(self, triplets):
        '''
        This function saves the sentence phrases into neo4j. It ensure creation of single nodes per phrase.
        This is achieved by first checking if the node is there in neo4j db based on phrase text and s_uuid.
        A new node is created only if the the node does not exist yet
        '''
        for triplet in triplets:
            log.debug(f"{triplet=}")

            head = self.G_n4j.nodes.match(C.PHRASE, name=triplet.head.phrase, s_uuid=self.sentence_uuid).first()
            if head is None:
                head = triplet.head
            tail = self.G_n4j.nodes.match(C.PHRASE, name=triplet.tail.phrase, s_uuid=self.sentence_uuid).first()
            if tail is None:
                tail = triplet.tail
            link_phrase = triplet.phrase
            if link_phrase == '':
                link_phrase = "-x-"

            log.debug(f"{head=}, {triplet.phrase=}, {tail=}")

            self.G_n4j.create(PhraseEdge(head, link_phrase , tail, self.sentence_uuid))

class PhraseNode(p2n.Node):
    def __init__(self, sentence_uuid, phrase, classification):
        '''
        Creates a Node which is specific to a Phrase. It automatically puts the neo4j Node's label as "Phrase"
        :param sentence_uuid: the unique id of the sentence
        :param phrase: this is the text that will show up on the graph Node
        :param classification: this is the type of phrase - Subject, Object, Attribute, etc.
        '''
        self.sentence_uuid = sentence_uuid
        self.phrase = phrase
        self.classification = classification
        # The Node is created with the values that are used during initialization. Even if the attributes are set, they are of no use per se.
        super().__init__(C.PHRASE, name=phrase, s_uuid=sentence_uuid, classification=classification)

class PhraseEdge(p2n.Relationship):
    def __init__(self, head, phrase, tail, sentence_uuid):
        '''
        Creates an Edge which is specific to a Phrase. It addds the Relationship property for classification as "Phrase_Link"
        :param head: The head PhraseNode
        :param tail: The tail PhraseNode
        :param sentence_uuid: the unique id of the sentence
        :param phrase: this is the text that will show up on the neo4j Relationship or link
        '''
        self.head = head
        self.phrase = phrase          
        self.tail = tail
        self.sentence_uuid = sentence_uuid
        # The Edge is created with the values that are used during initialization. Even if the attributes are set, they are of no use per se. 
        # Hence a new one needs to be created if you want different values. That's what is happening in the Sentence_Graph.save()
        super().__init__(head, phrase, tail, s_uuid=sentence_uuid, classification=C.PHRASE_LINK)

@dataclass
class KBNode:
    id : str
    phrase : str

@dataclass
class DBPediaNode (KBNode):
    source : str = C.DBPEDIA_TYPE

@dataclass
class NERNode:
    id : str
    phrase : str
    ner_type: str