#----------------------------#
# Author: Surjit Das
# Email: surjitdas@gmail.com
# Program: artmind
#----------------------------#

from loguru import logger as log
import spacy
from string import punctuation
import sys
from timefhuman import timefhuman as th
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from datetime import datetime
import ast
import external_kbs

# WIKIFIER_URL = "http://www.wikifier.org/annotate-article"
# WIKIDATA_API_ENDPOINT_URL = "https://www.wikidata.org/w/api.php"
# WIKIDATA_SPARQL_ENDPOINT_URL = "https://query.wikidata.org/sparql"
SQL_LOCAL_DB = "/Users/surjitdas/Downloads/nlu_processor/nlu_processor.db"
GRAPHML_PATH = "/Users/surjitdas/Downloads/nlu_processor/nlu_processor.graphml"
LOG_PATH = '/Users/surjitdas/Downloads/nlu_processor/nlu_processor.log'

NER = "NER"

ROOT = "ROOT"
SUBJECT = "SUBJECT"
PREDICATE = "PREDICATE"
OBJECT = "OBJECT"
COMPOUND = "COMPOUND"
MODIFIER = "MODIFIER"
SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "pobj", "dative", "oprd", "attr"] #attr - is an interesting one as Object, removing this makes attr as PREDICATE
COMPOUNDS = ["compound"]
MODIFIERS = ["amod", "advmod"]
EXCLUSIONS = ["det", "punct"]
# ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod", "ccomp", "complm",
#               "hmod", "infmod", "xcomp", "rcmod", "poss"," possessive"]
# PREPOSITIONS = ["prep"]
# "attr" removed from OBJECTS list. It is further qualifying a predicate or verb

# cn_l.connect(CONCEPTNET_LOCAL_DB)
log.add(sys.stderr, format="xx{time} {level} {message}", level="INFO")
log.add(LOG_PATH, format="{time} {level} {message}", level="INFO")

def plot_graph(G, title=None):
    # set figure size
    plt.figure(figsize=(10,10))
    
    # define position of nodes in figure
    pos = nx.nx_agraph.graphviz_layout(G)
    
    # draw nodes and edges
    nx.draw(G, pos=pos, with_labels=True)
    
    # get edge labels (if any)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    
    # draw edge labels (if any)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # plot the title (if any)
    plt.title(title)
    
    plt.show()
    return

# ----------------------------------------------------
# Class TextProcessor
# ----------------------------------------------------

class TextProcessor:
    def __init__(self, text):
        self.text = text
        log.debug(text)
        nlp = spacy.load("en_core_web_trf")
        self.doc = nlp(text)
        self.COLUMNS = ["TYPE", "NER.type",
                    "item","token.dep_","token.pos_","token.head.text","token.lemma_",
                    "compound_noun","verb_phrase",
                    "list_wdInstance","list_wikiDataClass", "list_dbPediaType","list_conceptNetType",
                    "ts"]
        self.db = sqlite3.connect(SQL_LOCAL_DB)
        self.G = nx.read_graphml(GRAPHML_PATH) # This line will throw an error if .graphml is not present
        self.kbs = external_kbs.Explorer()

    def add_meta_nodes(self, G, row, sources):
        for source in sources:
            label_list = ast.literal_eval(row[f"list_{source}"])
            for label in label_list[:3]:
                G.add_node(label, classification="EntityType", type=source)
                G.add_edge(row["item"],label,type=source)

    def process_tokens(self, sentence, text_df):
        # Add tokens to the dataframe - 1 row per token as the 2nd row onwards for the para/sentence
        log.debug("|token.text| token.dep_| token.pos_| token.head.text|token.lemma_|")
        for token in sentence:
            log.debug(f"|{token.text:<12}| {token.dep_:<10}| {token.pos_:<10}| {token.head.text:12}|{token.lemma_:12}")
            
            row = {"TYPE":"TOKEN","item":token.text,"token.dep_":token.dep_,"token.pos_":token.pos_,"token.head.text":token.head.text,"token.lemma_":token.lemma_}
            
            # Processing nouns
            if(token.pos_ in ['PROPN','NOUN']):
                row["list_wdInstance"] = str(self.kbs.get_wikidata(token.text)['list_wdInstance'])
                
                wk_dict = self.kbs.wikifier(token)
                row["list_wikiDataClass"] = str(wk_dict["list_wikiDataClass"])
                row["list_dbPediaType"] = str(wk_dict["list_dbPediaType"])
                
                if(token.pos_ == 'NOUN'):
                    row["list_conceptNetType"] = str(self.kbs.get_conceptnet_data(token.lemma_.lower()))
            # Processing compound nouns
            if(token.dep_ == 'compound'):
                row["compound_noun"] = f"{token.text} {token.head.text}"
            
            # Processing verb phrases
            if (token.dep_ == 'dobj') :
                row["verb_phrase"] = f"{token.head.text} {token.text}"
            
            row["ts"] = datetime.now()
            text_df = text_df.append(row, ignore_index=True)
        return text_df

    def algo1_execute(self):
        for sentence in self.doc.sents:
            spacy_data, text_df = self.algo1_process_sentence(self.db, sentence)
            self.algo1_create_graph(self.G, spacy_data, text_df)
        nx.write_graphml(self.G, GRAPHML_PATH)
        
        # for node in self.G.nodes(data=True):
        #     log.debug(node)
        # plot_graph(self.G)
        
        return

    def algo1_process_sentence(self, db, sentence):
        text_df = pd.DataFrame(columns=self.COLUMNS)
        # Add subject, predicate, object & NER to the dataframe - as 1st row for the para/sentence
        spacy_data = self.algo1_spacy_data(sentence)
        # row = {SUBJECT: str(spacy_data[SUBJECT]), PREDICATE: str(spacy_data[PREDICATE]), OBJECT: str(spacy_data[OBJECT]), NER: str(spacy_data[NER]), "ts" : datetime.now()}
        subject_row = {"TYPE":SUBJECT, "item": str(spacy_data[SUBJECT]), "ts" : datetime.now()}
        text_df = text_df.append(subject_row, ignore_index=True)
        predicate_row = {"TYPE":PREDICATE, "item": str(spacy_data[PREDICATE]), "ts" : datetime.now()}
        text_df = text_df.append(predicate_row, ignore_index=True)
        object_row = {"TYPE":OBJECT, "item": str(spacy_data[OBJECT]), "ts" : datetime.now()}
        text_df = text_df.append(object_row, ignore_index=True)
        for item in spacy_data[NER]:
            ner_row = {"TYPE":NER, "item":item, "NER.type":spacy_data[NER][item], "ts" : datetime.now()}
            ner_row["list_wdInstance"] = str(self.kbs.get_wikidata(item)['list_wdInstance'])
            wk_dict = self.kbs.wikifier(item)
            ner_row["list_wikiDataClass"] = str(wk_dict["list_wikiDataClass"])
            ner_row["list_dbPediaType"] = str(wk_dict["list_dbPediaType"])
            text_df = text_df.append(ner_row, ignore_index=True)

        text_df = self.process_tokens(sentence, text_df)
        
        # Write the text_df to db
        text_df.to_sql("paragraph", db, if_exists="append")

        return spacy_data, text_df

    def algo1_create_graph(self, G, spacy_data, text_df):
        ...
        '''
        Driving loops are as follows; each one checks if it was processed in the earlier loop or not
        NERs
        Compound Nouns ?
        Proper Nouns
        Verb Phrases
        Nouns
        Pronouns or Dets?
        SUBJECT
        OBJECT
        PREDICATE - AS LINKS
        '''
        for index, row in text_df.iterrows():
            
            if row["TYPE"] == NER:
                G.add_node(row["item"], classification="Entity", type=row["NER.type"])
                
                self.add_meta_nodes(G, row, ["wdInstance","wikiDataClass","dbPediaType"])

            if str(row["compound_noun"]) != 'nan':
                pass # This is a ToDo ... Decide whether compound nouns need to be handled & whether this is the right place for this code

            if (row["TYPE"] == "TOKEN" and row["token.pos_"] in ["PROPN","NOUN"] and (row["item"] not in spacy_data["NER"])):
                G.add_node(row["item"], classification="Entity")
                
                self.add_meta_nodes(G, row, ["wdInstance","wikiDataClass","dbPediaType"])

                if (row["token.pos_"] == "NOUN") and str(row["list_conceptNetType"]) != 'nan':
                    self.add_meta_nodes(G, row, ["conceptNetType"])
                
            if(row["token.pos_"] == "PRON"):
                G.add_node(row["item"], classification="Entity", type="EntityPointer")

            if str(row["verb_phrase"]) != 'nan':
                G.add_node(row["verb_phrase"], classification="Activity")
    
    # My own function invented to create the best chunks out of the sentences
    def algo1_spacy_data(self, doc):
        subject = []
        subject_words = []
        predicate = []
        object_ = []
        
        log.debug("|chunk.text|chunk.root|chunk.root.dep_|")
        for chunk in doc.noun_chunks:
            log.debug(f"|{chunk.text:30}|{chunk.root.text:12}|{chunk.root.dep_:10}|")
            if 'subj' in chunk.root.dep_:
                subject.append(chunk.text)
                subject_words = chunk.text.split()
            elif 'obj' in chunk.root.dep_:
                object_.append(chunk.text)
            else:
                predicate.append(chunk.text)
        
        log.debug(f"Post noun chunks... subject:{subject}; subject_words:{subject_words}; predicate:{predicate}; object_:{object_}")

        for token in doc:
            if 'ROOT' == token.dep_ and token.pos_ != 'AUX':
                predicate.append(token.text)
                log.debug(f"Root predicate: {predicate}")
                for child in token.children:
                    log.debug(f"Processing child: {child.text}; subject: {subject}; predicate{predicate}")
                    if child.text not in subject_words and child.text not in predicate and child.pos_ not in ['ADP', 'AUX', 'PUNCT']:
                        predicate.append(child.text)
                        log.debug(f"predicate after child: {predicate}")
            elif token.dep_ in ['advcl', 'advmod', 'oprd', 'xcomp'] and (token.text not in subject_words and token.text not in predicate):
                predicate.append(token.text)
                log.debug(f"Non-root predicate: {predicate}")

        ner_dict = {}
        for ent in doc.ents:
            ner_dict[ent.text] = ent.label_

        spacy_data = {
                SUBJECT:subject,
                PREDICATE:predicate,
                OBJECT:object_,
                NER: ner_dict
                }
        
        log.debug(spacy_data)
        return spacy_data
    
    def algo2_get_keytype(self, dep):
        key = ""
        if dep == ROOT:
            key = ROOT
        elif dep in SUBJECTS:
            key = SUBJECT
        elif dep in OBJECTS:
            key = OBJECT
        elif dep in COMPOUNDS:
            key = COMPOUND
        elif dep in MODIFIERS:
            key = MODIFIER
        else:
            key = PREDICATE
        return key

def main():
    text=input("Para: ")
    while(text!="/stop"):
        tp = TextProcessor(text)
        tp.algo1_execute()
        print("Done...")
        text = input("Para: ")

if __name__=="__main__":
    main()