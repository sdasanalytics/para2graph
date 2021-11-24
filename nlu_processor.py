#----------------------------#
# Author: Surjit Das
# Email: surjitdas@gmail.com
# Program: artmind
#----------------------------#

import logging as log
import spacy
import urllib
from string import punctuation
import json
import sys
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from timefhuman import timefhuman as th
import networkx as nx
import matplotlib.pyplot as plt
import conceptnet_lite as cn_l
import pandas as pd
import sqlite3
from datetime import datetime
import ast

SUBJECT = "SUBJECT"
PREDICATE = "PREDICATE"
OBJECT = "OBJECT"
NER = "NER"

WIKIFIER_URL = "http://www.wikifier.org/annotate-article"
WIKIDATA_API_ENDPOINT_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_ENDPOINT_URL = "https://query.wikidata.org/sparql"
# CONCEPTNET_API_ENDPOINT_URL = "http://api.conceptnet.io/c/en/" not using the Web API, but directly the local database & API
CONCEPTNET_LOCAL_DB = "/Volumes/Surjit_SSD_1/tech/conceptnet.db"
SQL_LOCAL_DB = "/Users/surjitdas/Downloads/nlu_processor.db"
GRAPHML_PATH = "/Users/surjitdas/Downloads/nlu_processor.graphml"
LOG_PATH = '/Users/surjitdas/Downloads/nlu_processor.log'

cn_l.connect(CONCEPTNET_LOCAL_DB)
log.basicConfig(level=log.INFO, filename=LOG_PATH, filemode='w', format='%(message)s')

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

    def execute(self):
        for sentence in self.doc.sents:
            spacy_data, text_df = self.process_sentence(self.db, sentence)
            self.create_graph(self.G, spacy_data, text_df)
        
        for node in self.G.nodes(data=True):
            log.info(node)
        # plot_graph(self.G)
        nx.write_graphml(self.G, GRAPHML_PATH)
        return

    def process_sentence(self, db, sentence):
        text_df = pd.DataFrame(columns=self.COLUMNS)
        # Add subject, predicate, object & NER to the dataframe - as 1st row for the para/sentence
        spacy_data = self.get_spacy_data(sentence)
        # row = {SUBJECT: str(spacy_data[SUBJECT]), PREDICATE: str(spacy_data[PREDICATE]), OBJECT: str(spacy_data[OBJECT]), NER: str(spacy_data[NER]), "ts" : datetime.now()}
        subject_row = {"TYPE":SUBJECT, "item": str(spacy_data[SUBJECT]), "ts" : datetime.now()}
        text_df = text_df.append(subject_row, ignore_index=True)
        predicate_row = {"TYPE":PREDICATE, "item": str(spacy_data[PREDICATE]), "ts" : datetime.now()}
        text_df = text_df.append(predicate_row, ignore_index=True)
        object_row = {"TYPE":OBJECT, "item": str(spacy_data[OBJECT]), "ts" : datetime.now()}
        text_df = text_df.append(object_row, ignore_index=True)
        for item in spacy_data[NER]:
            ner_row = {"TYPE":NER, "item":item, "NER.type":spacy_data[NER][item], "ts" : datetime.now()}
            ner_row["list_wdInstance"] = str(self.get_wikidata(item)['list_wdInstance'])
            wk_dict = self.wikifier(item)
            ner_row["list_wikiDataClass"] = str(wk_dict["list_wikiDataClass"])
            ner_row["list_dbPediaType"] = str(wk_dict["list_dbPediaType"])
            text_df = text_df.append(ner_row, ignore_index=True)

        # Add tokens to the dataframe - 1 row per token as the 2nd row onwards for the para/sentence
        log.debug("|token.text| token.dep_| token.pos_| token.head.text|token.lemma_|")
        for token in sentence:
            log.debug(f"|{token.text:<12}| {token.dep_:<10}| {token.pos_:<10}| {token.head.text:12}|{token.lemma_:12}")
            
            row = {"TYPE":"TOKEN","item":token.text,"token.dep_":token.dep_,"token.pos_":token.pos_,"token.head.text":token.head.text,"token.lemma_":token.lemma_}
            
            # Processing nouns
            if(token.pos_ in ['PROPN','NOUN']):
                row["list_wdInstance"] = str(self.get_wikidata(token.text)['list_wdInstance'])
                
                wk_dict = self.wikifier(token)
                row["list_wikiDataClass"] = str(wk_dict["list_wikiDataClass"])
                row["list_dbPediaType"] = str(wk_dict["list_dbPediaType"])
                
                if(token.pos_ == 'NOUN'):
                    row["list_conceptNetType"] = str(self.get_conceptnet_data(token.lemma_.lower()))
            # Processing compound nouns
            if(token.dep_ == 'compound'):
                row["compound_noun"] = f"{token.text} {token.head.text}"
            
            # Processing verb phrases
            if (token.dep_ == 'dobj') :
                row["verb_phrase"] = f"{token.head.text} {token.text}"
            
            row["ts"] = datetime.now()
            text_df = text_df.append(row, ignore_index=True)
        
        # Write the text_df to db
        text_df.to_sql("paragraph", db, if_exists="append")

        return spacy_data, text_df

    def create_graph(self, G, spacy_data, text_df):
        ...
        '''
        Driving loops are as follows; each one checks if it was processed in the earlier loop or not
        NERs
        Compound Nouns ?
        Proper Nouns
        Verb Phrases
        Nouns
        Pronouns or Dets?
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
    

    def add_meta_nodes(self, G, row, sources):
        for source in sources:
            label_list = ast.literal_eval(row[f"list_{source}"])
            for label in label_list[:3]:
                G.add_node(label, classification="EntityType", type=source)
                G.add_edge(row["item"],label,type=source)

        
    # My own function invented to create the best chunks out of the sentences
    def get_spacy_data(self, doc):
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
                    if child.text not in subject_words and child.text not in predicate and child.pos_ not in ['ADP', 'AUX']:
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
    
    def wikifier(self, text, lang="en", threshold=0.8):
        """Function that fetches entity linking results from wikifier.com API"""
        # Prepare the URL.
        data = urllib.parse.urlencode([
            ("text", text), ("lang", lang),
            ("userKey", "vvswrnlywccfgddhmprbdwviamhnuc") # this is my userkey
            ,
            ("pageRankSqThreshold", "%g" %
            threshold), ("applyPageRankSqThreshold", "false"), # This flag is important - true seems to filter out a lot
            ("nTopDfValuesToIgnore", "100"), ("nWordsToIgnoreFromList", "100"),
            ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
            ("support", "true"), ("ranges", "false"), ("minLinkFrequency", "2"),
            ("includeCosines", "false"), ("maxMentionEntropy", "3")
        ])
    
        # Call the Wikifier and read the response.
        req = urllib.request.Request(WIKIFIER_URL, data=data.encode("utf8"), method="POST")
        with urllib.request.urlopen(req, timeout=60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))
        
        log.debug(data)
        log.debug(response)
            # The response is of the following JSON format
            # {'annotations': 
            #     [
            #         {
            #             'title': 'Diwali', 'url': 'http://en.wikipedia.org/wiki/Diwali', ...
            #             'wikiDataClasses': [
            #                                 {'itemId': 'Q132241', 'enLabel': 'festival'},
            #                                 {'itemId': 'Q1445650', 'enLabel': 'holiday'},
            #                                 {'itemId': 'Q15275719', 'enLabel': 'recurring event'}, ...
            #                                 ],
            #             'dbPediaTypes': ['Holiday'],
            #             'dbPediaIri': 'http://dbpedia.org/resource/Apple', ...
            #         }
            #     ],
            #     'spaces': ['', ' ', ''],
            #     'words': ['Diwali'],...
            # }

        # results = filter_wikifier_response(response)
        wk_dict = {"list_wikiDataClass":[],"list_dbPediaType":[]}
        for record in response["annotations"]:
            # print(record)
            wdClassList = []
            wikiDataClasses = record['wikiDataClasses']
            for item in wikiDataClasses:
                wdClassList.append(item['enLabel'])
            dbpTypeList = record['dbPediaTypes']

            wk_dict['list_wikiDataClass'].extend(wdClassList)
            wk_dict['list_dbPediaType'].extend(dbpTypeList)

        return wk_dict

    # For wikidata, defining a generic sparql calling function
    def get_sparql_results(self, query):
        user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
        # TODO adjust user agent; see https://w.wiki/CX6
        sparql = SPARQLWrapper(WIKIDATA_SPARQL_ENDPOINT_URL, agent=user_agent)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        return sparql.query().convert()

    def get_wikidata(self, text, limit=1):

        # First the web serach api of wiki data has to be used for getting the entity id
        # This entity id is then useful for the wikidata query service
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': 'en',
            'search': text,
            'limit': limit
        }
        response = requests.get(WIKIDATA_API_ENDPOINT_URL, params=params)
        response_json = response.json()
        log.debug(f"get_wikidata::response_json: {response_json}")
        entity_ids = []
        for entity in response_json['search']:
            entity_ids.append(entity['id'])
        
        records_dict = {}
        # Put an empty list if wd does not find the entity
        if len(entity_ids)==0:
            records_dict["list_wdInstance"] = ["wd_UNKNOWN"]
            return records_dict

        entity_id = entity_ids[0] # First entity_id. If there are none, this will throw an error
        query =(f"SELECT ?instance_of ?instance_ofLabel "
                f"?subclass_of ?subclass_ofLabel "
                f"WHERE {{wd:{entity_id} "
                f"wdt:P31 ?instance_of . "
                f"?instance_of wdt:P279+ ?subclass_of . "
                f"SERVICE wikibase:label {{ bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. }}}}")
        log.debug(query)
        results = self.get_sparql_results(query)
        log.debug(results)
        records = (results["results"]["bindings"])
        
        for record in records:
            column1 = record['instance_ofLabel']['value']
            column2 = record['subclass_ofLabel']['value']
            col2_list = records_dict.get(column1,[])
            col2_list.append(column2)
            records_dict[column1] = col2_list
        
        records_dict["list_wdInstance"] = list(records_dict.keys())
        
        log.debug(records_dict)
        return records_dict

    def get_conceptnet_data(self, text):
        conceptnet_list = []
        try:
            for e in cn_l.edges_for(cn_l.Label.get(text=text, language='en').concepts, same_language=True):
                if(e.relation.name in ['is_a'] and e.start.text == text):
                    conceptnet_list.append(e.end.text)
        except:
            conceptnet_list.append("UNKNOWN")
        return conceptnet_list

def main():
    text=input("Para: ")
    while(text!="/stop"):
        tp = TextProcessor(text)
        tp.execute()
        print("Done...")
        text = input("Para: ")

if __name__=="__main__":
    main()