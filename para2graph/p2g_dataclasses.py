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
from datetime import datetime
import pandas as pd

class SentenceGraph:
    def __init__(self, G_n4j, sentence_uuid) -> None:
        self.G_n4j = G_n4j
        self.sentence_uuid = sentence_uuid

    def save(self, ph_3plets, noun_ner_3plets, kb3_plets):
        '''
        This function saves the sentence phrases into neo4j. It ensure creation of single nodes per phrase.
        This is achieved by first checking if the node is there in neo4j db based on phrase text and s_uuid.
        A new node is created only if the the node does not exist yet
        '''
        for ph_3plet in ph_3plets:
            log.debug(f"{ph_3plet=}")

            head = self.G_n4j.nodes.match(C.PHRASE, name=ph_3plet.head.phrase, s_uuid=self.sentence_uuid).first()
            if head is None:
                head = ph_3plet.head
            tail = self.G_n4j.nodes.match(C.PHRASE, name=ph_3plet.tail.phrase, s_uuid=self.sentence_uuid).first()
            if tail is None:
                tail = ph_3plet.tail
            link_phrase = ph_3plet.phrase
            if link_phrase == '':
                link_phrase = "-x-"

            log.debug(f"{head=}, {ph_3plet.phrase=}, {tail=}")
            phrase_edge = PhraseEdge(head, link_phrase , tail, self.sentence_uuid)
            self.G_n4j.create(phrase_edge)

        for noun_ner_3plet in noun_ner_3plets:
            log.debug(f"{noun_ner_3plet=}")

            head = self.G_n4j.nodes.match(C.PHRASE, name=noun_ner_3plet.head.phrase, s_uuid=self.sentence_uuid).first()
            if head is None:
                head = noun_ner_3plet.head
            tail = self.G_n4j.nodes.match(noun_ner_3plet.tail.type, name=noun_ner_3plet.tail.text).first()
            if tail is None:
                tail = noun_ner_3plet.tail

            log.debug(f"{head=}, {tail=}")
            edge = PhraseInfoEdge(head, tail)
            self.G_n4j.create(edge)

        for kb3_plet in kb3_plets:
            log.debug(f"{kb3_plet=}")

            head = self.G_n4j.nodes.match(kb3_plet.head.type, name=kb3_plet.head.text).first()
            if head is None:
                head = kb3_plet.head
            tail = self.G_n4j.nodes.match(kb3_plet.tail.type, name=kb3_plet.tail.text).first()
            if tail is None:
                tail = kb3_plet.tail

            log.debug(f"{head=}, {tail=}")
            edge = PhraseInfoEdge(head, tail)
            self.G_n4j.create(edge)            


class SentenceTable:
    def __init__(self, db) -> None:
        self.db = db

    def persist(self, sentence_uuid, sentence):
        '''
        This function saves the sentence tokens along with the token information, as well as NERs to the database
        '''        
        nouns = []
        ners = []
        log.debug("|token.text| token.dep_| token.pos_| token.head.text|token.lemma_|")
        df = pd.DataFrame()
        for token in sentence:
            log.debug(f"|{token.text:<12}| {token.dep_:<10}| {token.pos_:<10}| {token.head.text:12}|{token.lemma_:12}")
            row = {C.COL_SENT_UUID:sentence_uuid, C.COL_TYPE:C.COL_TYPE_VAL_TOKEN, C.COL_ITEM:token.text, 
                    C.COL_TOKEN_DEP:token.dep_, C.COL_TOKEN_POS:token.pos_, C.COL_TOKEN_HEAD_TEXT:token.head.text, C.COL_TOKEN_LEMMA:token.lemma_,
                    C.COL_TS: datetime.now()}
            df = df.append(row, ignore_index=True)
            if token.pos_ in [C.POS_NOUN, C.POS_PROPER_NOUN]:
                nouns.append(token.text)

        for entity in sentence.ents:
            ner_row = {C.COL_SENT_UUID:sentence_uuid, C.COL_TYPE:C.NER, C.COL_ITEM:entity.text, C.COL_NER_TYPE:entity.label_, C.COL_TS:datetime.now()}
            df = df.append(ner_row, ignore_index=True)
            ners.append([entity.text, entity.label_])
        
        df.to_sql(C.TAB_SENTENCES, self.db, if_exists="append", index=False)

        return nouns, ners

class ExternalKBsTable:
    ...

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

    def __eq__(self, other):
        '''
        Equals if name=phrase & sentencue_uuid matches. Ignored classification.
        '''
        if (self.sentence_uuid == other.sentence_uuid and self.phrase == other.phrase):
            return True
        else:
            return False

    def __hash__(self):
        return super().__hash__()

class PhraseEdge(p2n.Relationship):
    def __init__(self, head, phrase, tail, sentence_uuid):
        '''
        Creates an Edge which is specific to a Phrase. It adds the Relationship property for classification as "Phrase_Link"
        :param head: The head PhraseNode
        :param tail: The tail PhraseNode
        :param sentence_uuid: the unique id of the sentence
        :param phrase: this is the text that will show up on the neo4j Relationship or link
        '''
        self.head = head
        self.phrase = phrase # neo4j relationship type 
        self.tail = tail
        self.sentence_uuid = sentence_uuid
        # The Edge is created with the values that are used during initialization. Even if the attributes are set, they are of no use per se. 
        # Hence a new one needs to be created if you want different values. That's what is happening in the Sentence_Graph.save()
        super().__init__(head, phrase, tail, s_uuid=sentence_uuid, classification=C.PHRASE_LINK)

class NounNode(p2n.Node):
    def __init__(self, noun_text):
        '''
        Creates a Node which is specific to a Noun. It automatically puts the neo4j Node's label as "Noun"
        :param noun_text: this is the text that will show up on the graph Node
        '''
        self.text = noun_text
        self.type = C.NOUN
        # The Node is created with the values that are used during initialization. Even if the attributes are set, they are of no use per se.
        super().__init__(C.NOUN, name=noun_text)

class NERNode(p2n.Node):
    def __init__(self, ner_text, ner_type):
        '''
        Creates a Node which is specific to a NER. It automatically puts the neo4j Node's label as the NER type
        :param ner_text: this is the text that will show up on the graph Node
        :param ner_type: this is the label of the n4j Node
        '''
        self.text = ner_text
        self.type = ner_type
        # The Node is created with the values that are used during initialization. Even if the attributes are set, they are of no use per se.
        super().__init__(ner_type, name=ner_text)

class PhraseInfoEdge(p2n.Relationship):
    def __init__(self, head, tail):
        '''
        Creates an Edge which is specific to a Phrase. 
        :param head: The head PhraseNode
        :param tail: The tail NER or Noun Node
        '''
        self.head = head
        self.tail = tail
        # The Edge is created with the values that are used during initialization. Even if the attributes are set, they are of no use per se. 
        # Hence a new one needs to be created if you want different values. That's what is happening in the Sentence_Graph.save()
        super().__init__(head, "-", tail)

class KBNode(p2n.Node):
    def __init__(self, text, kb_source):
        '''
        Creates a Node which is specific to a KB. It automatically puts the neo4j Node's label as kb_source
        :param text: this is the text that will show up on the graph Node
        :param kb_source: this is the Node label
        '''
        self.text = text
        self.type = kb_source
        # The Node is created with the values that are used during initialization. Even if the attributes are set, they are of no use per se.
        super().__init__(kb_source, name=text)

# class InfoKBEdge(p2n.Relationship):
#     def __init__(self, head, tail):
#         '''
#         Creates an Edge which is specific to a Phrase. 
#         :param head: The head NER or Noun Node 
#         :param tail: The KB Node
#         '''
#         self.head = head
#         self.tail = tail
#         # The Edge is created with the values that are used during initialization. Even if the attributes are set, they are of no use per se. 
#         # Hence a new one needs to be created if you want different values. That's what is happening in the Sentence_Graph.save()
#         super().__init__(head, "-", tail)

