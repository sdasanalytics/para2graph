#----------------------------#
# Author: Surjit Das
# Email: surjitdas@gmail.com
# Program: artmind
#----------------------------#

from copy import Error
from loguru import logger as log
from networkx.generators.small import heawood_graph
from networkx.readwrite.gexf import GEXF
import spacy
from string import punctuation
import sys
from spacy.util import raise_error
from timefhuman import timefhuman as th
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import uuid
from datetime import datetime
import ast
import external_kbs
import openpyxl

SQL_LOCAL_DB = "/Users/surjitdas/Downloads/nlu_processor/nlu_processor_v2.db"
GEXF_PATH = "/Users/surjitdas/Downloads/nlu_processor/nlu_processor.gexf"
GRAPHML_PATH = "/Users/surjitdas/Downloads/nlu_processor/nlu_processor.graphml"
LOG_PATH = '/Users/surjitdas/Downloads/nlu_processor/nlu_processor.log'


log.remove() #removes default handlers
log.add(LOG_PATH, format="{time} {level} {message}", level="DEBUG")

NER = "NER"
SPO = "SPO"

ROOT = "ROOT"
SUBJECT = "SUBJECT"
PREDICATE = "PREDICATE"
OBJECT = "OBJECT"
COMPOUND = "COMPOUND"
MODIFIER = "MODIFIER"
SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "pobj", "dative", "oprd", "attr"] #attr - is an interesting one as Object, removing this makes attr as PREDICATE
COMPOUNDS = ["compound"]
MODIFIERS = ["amod", "advmod", "nummod", "npadvmod"]
EXCLUSIONS = ["det", "punct"]
# ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod", "ccomp", "complm",
#               "hmod", "infmod", "xcomp", "rcmod", "poss"," possessive"]
# PREPOSITIONS = ["prep"]
# "attr" removed from OBJECTS list. It is further qualifying a predicate or verb

ENTITY = "Entity"
LINK = "Link"
ATTRIBUTE = "Attribute"
ACTIVITY = "Activity"
LABEL = "Label"
LINK_LABEL = "Link_Label"
CLASSIFICATION = "Classification"
TYPE = "Type"
NOUN = "Noun"

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
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.COLUMNS = ["sentence_uuid", "TYPE", "NER_type",
                    "item","token_dep","token_pos","token_head_text","token_lemma",
                    "compound_noun","verb_phrase",
                    "list_wdInstance","list_wikiDataClass", "list_dbPediaType","list_conceptNetType",
                    "ts"]
        self.COLUMNS_PARA = ["sentence_uuid", "TYPE", "NER_type",
                    "item","token_dep","token_pos","token_head_text","token_lemma",
                    "compound_noun","verb_phrase",
                    "ts"]
        self.COLUMNS_KBS = ["item",
                    "list_wdInstance","list_wikiDataClass", "list_dbPediaType","list_conceptNetType",
                    "ts"]                                        
        self.db = sqlite3.connect(SQL_LOCAL_DB)
        # self.G = nx.read_gexf(GEXF_PATH) # This line will throw an error if .gexf is not present
        self.G = nx.MultiDiGraph()
        self.kbs = external_kbs.Explorer()

    def process_save_ners_tokens(self, sentence_uuid, sentence, parsed_context):
        text_df = pd.DataFrame(columns=self.COLUMNS)
        parsed_context_row = {"sentence_uuid":sentence_uuid, "TYPE":"PARSED", "item": str(parsed_context), "ts" : datetime.now()}
        text_df = text_df.append(parsed_context_row, ignore_index=True)
        for ent in sentence.ents:
            ner_row = {"sentence_uuid":sentence_uuid, "TYPE":NER, "item":ent.text, "NER_type":ent.label_, "ts" : datetime.now()}
            sql_str = f"select * from external_kbs where item=?"
            params = (ent.text,)
            df = pd.read_sql(sql_str, self.db, params=params)
            if len(df) == 0:
                ner_row["list_wdInstance"] = str(self.kbs.get_wikidata(ent.text)['list_wdInstance'])
                wk_dict = self.kbs.wikifier(ent.text)
                ner_row["list_wikiDataClass"] = str(wk_dict["list_wikiDataClass"])
                ner_row["list_dbPediaType"] = str(wk_dict["list_dbPediaType"])
                
                kbs_dict = {"item":[ent.text], 
                            "list_wdInstance": [ner_row["list_wdInstance"]], "list_wikiDataClass": [ner_row["list_wikiDataClass"]], "list_dbPediaType": [ner_row["list_dbPediaType"]],
                            "ts" : [datetime.now()]}
                kbs_df = pd.DataFrame(kbs_dict)
                kbs_df.to_sql("external_kbs", self.db, index=False, if_exists="append")
            else:
                ner_row["list_wdInstance"] = df["list_wdInstance"][0]
                ner_row["list_wikiDataClass"] = df["list_wikiDataClass"][0]
                ner_row["list_dbPediaType"] = df["list_dbPediaType"][0]
            
            text_df = text_df.append(ner_row, ignore_index=True)

        text_df = self.process_tokens(sentence_uuid, sentence, text_df)
        
        # Write the text_df to db
        para_df = text_df[self.COLUMNS_PARA]
        para_df.to_sql("sentences", self.db, index=False, if_exists="append")
        return text_df

    def add_meta_nodes(self, G, row, sources):
        for source in sources:
            label_list = ast.literal_eval(row[f"list_{source}"])
            for label in label_list[:3]:
                if label not in ['Wikimedia disambiguation page', 'MediaWiki main-namespace page', 'list', 'word-sense disambiguation', 'Wikimedia internal item', 'MediaWiki page', 'wd_UNKNOWN']:
                    head = row["item"]
                    tail = (label, {CLASSIFICATION:"EntityType", TYPE:source})
                    G.add_nodes_from([tail])
                    link = (head,label,{TYPE:source})
                    G.add_edges_from([link])

    def algo1_execute(self, text):
        log.debug(f"Processing text: {text}")
        doc = self.nlp(text)
        for sentence in doc.sents:
            sentence_uuid = str(uuid.uuid4())
            spacy_data, text_df = self.algo1_process_sentence(sentence_uuid, sentence)
            self.algo1_create_graph(self.G, spacy_data, text_df)
        nx.write_gexf(self.G, GEXF_PATH)
        
        # for node in self.G.nodes(data=True):
        #     log.debug(node)
        # plot_graph(self.G)
        
        return

    def algo1_process_sentence(self, sentence_uuid, sentence):
        # Add subject, predicate, object & NER to the dataframe - as 1st row for the para/sentence
        spo_data = self.algo1_spacy_data(sentence)
        text_df = self.process_save_ners_tokens(sentence_uuid, sentence, spo_data)
        return spo_data, text_df

    def process_tokens(self, sentence_uuid, sentence, text_df):
        # Add tokens to the dataframe - 1 row per token as the 2nd row onwards for the para/sentence
        log.debug("|token.text| token.dep_| token.pos_| token.head.text|token.lemma_|")
        for token in sentence:
            log.debug(f"|{token.text:<12}| {token.dep_:<10}| {token.pos_:<10}| {token.head.text:12}|{token.lemma_:12}")
            sql_str = f"select * from external_kbs where item=?"
            params = (token.text,)
            df = pd.read_sql(sql_str, self.db, params=params)
            row = {"sentence_uuid":sentence_uuid, "TYPE":"TOKEN","item":token.text,"token_dep":token.dep_,"token_pos":token.pos_,"token_head_text":token.head.text,"token_lemma":token.lemma_}
            
            # Processing nouns
            if(token.pos_ in ['PROPN','NOUN']):
                if len(df) == 0:
                    row["list_wdInstance"] = str(self.kbs.get_wikidata(token.text)['list_wdInstance'])
                    
                    wk_dict = self.kbs.wikifier(token)
                    row["list_wikiDataClass"] = str(wk_dict["list_wikiDataClass"])
                    row["list_dbPediaType"] = str(wk_dict["list_dbPediaType"])
                    kbs_dict = {"item":[token.text], 
                            "list_wdInstance": [row["list_wdInstance"]], "list_wikiDataClass": [row["list_wikiDataClass"]], "list_dbPediaType": [row["list_dbPediaType"]],
                            "ts" : [datetime.now()]}
                    if(token.pos_ == 'NOUN'):
                        row["list_conceptNetType"] = str(self.kbs.get_conceptnet_data(token.lemma_.lower()))
                        kbs_dict["list_conceptNetType"] = [row["list_conceptNetType"]]

                    kbs_df = pd.DataFrame(kbs_dict)
                    kbs_df.to_sql("external_kbs", self.db, index=False, if_exists="append")
                else:
                    row["list_wdInstance"] = df["list_wdInstance"][0]
                    row["list_wikiDataClass"] = df["list_wikiDataClass"][0]
                    row["list_dbPediaType"] = df["list_dbPediaType"][0]
                    row["list_conceptNetType"] = df["list_dbPediaType"][0]
                    
            # Processing compound nouns
            if(token.dep_ == 'compound'):
                row["compound_noun"] = f"{token.text} {token.head.text}"
            
            # Processing verb phrases
            if (token.dep_ == 'dobj') :
                row["verb_phrase"] = f"{token.head.text} {token.text}"
            
            row["ts"] = datetime.now()
            text_df = text_df.append(row, ignore_index=True)
        return text_df

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
                G.add_node(row["item"], Classification="Entity", Type=row["NER_type"])
                
                self.add_meta_nodes(G, row, ["wdInstance","wikiDataClass","dbPediaType"])

            if str(row["compound_noun"]) != 'nan':
                pass # This is a ToDo ... Decide whether compound nouns need to be handled & whether this is the right place for this code

            if (row["TYPE"] == "TOKEN" and row["token_pos"] in ["PROPN","NOUN"] and (row["item"] not in spacy_data["NER"])):
                G.add_node(row["item"], Classification="Entity")
                
                self.add_meta_nodes(G, row, ["wdInstance","wikiDataClass","dbPediaType"])

                if (row["token_pos"] == "NOUN") and str(row["list_conceptNetType"]) != 'nan':
                    self.add_meta_nodes(G, row, ["conceptNetType"])
                
            if(row["token_pos"] == "PRON"):
                G.add_node(row["item"], Classification="Entity", Type="EntityPointer")

            if str(row["verb_phrase"]) != 'nan':
                G.add_node(row["verb_phrase"], Classification="Activity")
    
    # My own function invented to create the best chunks out of the sentences
    def algo1_spacy_data(self, doc):
        subject = []
        subject_words = []
        predicate = []
        object_ = []
        
        log.debug("|chunk.text|chunk.root|chunk.root.dep_|")
        for chunk in doc.noun_chunks:
            log.debug(f"|{chunk.text:30}|{chunk.root.text:12}|{chunk.root.dep_:10}|")
            # log.debug(dir(chunk))
            if 'subj' in chunk.root.dep_:
                subject.append(chunk.text)
                subject_words = chunk.text.split()
            elif 'obj' in chunk.root.dep_:
                if 'dobj' in chunk.root.dep_:
                    object_.append(f"{chunk.text} {chunk.root.text}")
                else:
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
                OBJECT:object_
                ,
                NER: ner_dict
                }
        
        log.debug(spacy_data)
        return spacy_data
    
    def breakit(self, item):
        key = ""
        value = ""
        for x, y in item.items():
            key = x
            value = y
        return key, value

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

    def algo2_execute(self, text):
        log.debug(f"Processing text: {text}")
        doc = self.nlp(text)
        for sentence in doc.sents:
            log.debug(f"sentence=")
            sentence_uuid = str(uuid.uuid4())
            spo_data, subject_data = self.algo2_process_sentence(sentence_uuid, sentence)
            sql_str = "select * from vw_sentences where sentence_uuid = ?"
            params = (sentence_uuid, )
            vw_text_df = pd.read_sql(sql_str, self.db, params=params)
            # log.debug(f"{vw_text_df=}")
            # log.debug(f"{str(vw_text_df['item'])=}")
            # self.algo1_create_graph(self.G, spacy_data, text_df)
        # nx.write_graphml(self.G, GRAPHML_PATH)

    def algo2_process_sentence(self, sentence_uuid, sentence):
        # Add subject, predicate, object & NER to the dataframe - as 1st row for the para/sentence
        context, spo_data, subject_data = self.algo2_spo_data(sentence)
        text_df = self.process_save_ners_tokens(sentence_uuid, sentence, [context, spo_data, subject_data])
        return spo_data, subject_data

    def algo2_spo_data(self, sentence):
        context=[{"ROOT":"x"}]
        for token in sentence:
            if (token.dep_ == "ROOT"):
                log.debug(f"{token.text} is the root")
                context = self.algo2_sentence_dfs(token,context=[{"ROOT":token.text}])
        log.debug(f"Final context: {context}")
        # log.debug(f"{context[0]=}")
        
        key0, value0 = self.breakit(context[0])
        key2, value2 = self.breakit(context[2])
        context_y = [context[1]]
        if key2 == PREDICATE:
            value2 = f"{value0} {value2}"
            context_y.append({PREDICATE:value2})
            context_y.extend(i for i in context[3:])
        else:
            context_y.append({PREDICATE:value0})
            context_y.extend(i for i in context[2:])
        log.debug(context_y)
        
        subject_q = []
        spo_list = []
        for item in context_y:
            key, value = self.breakit(item)
            if key == SUBJECT:
                subject_q.append(value)
                spo_list.append([value])
            elif key == PREDICATE:
                spo_list[-1].append(value)
            elif key == OBJECT:
                spo_list[-1].append(value)
                spo_list.append([subject_q[-1]])

        log.debug(f"{spo_list[:-1]=}")
        log.debug(f"{subject_q=}")
        return context, spo_list[:-1], subject_q

    def algo2_sentence_dfs(self, token, context):
        item = {}
        key = self.algo2_get_keytype(token.dep_)
        for x,y in context[-1].items():
            last_key = x
            last_value = y
        log.debug(f"{key=}, {token.text=}, {context=}, {last_key=}:{last_value=}")
        if key == ROOT:
            log.debug("Skipping as key is ROOT...")
        elif last_key != key:
            if (key == COMPOUND): 
                value = f"{token.text} {last_value}"
                log.debug(f"COMPOUND : {last_key}:{value}")
                context[-1] = {last_key:value}
            elif (key == MODIFIER and token in token.head.lefts):
                value = f"{token.text} {last_value}"
                log.debug(f"MODIFIER : {last_key}:{value}")
                context[-1] = {last_key:value}
            else:
                if key == MODIFIER:
                    key = PREDICATE
                item[key] = token.text
                log.debug(f"context.append({item=})")
                context.append(item)
        else: # if last_key == key
            value = f"{last_value} {token.text}"
            log.debug(f"if last_key == key: {key=}:{value=}")
            context[-1] = {key:value}
        log.debug(f"context after adding: {context}")

        for child in token.children:
            log.debug(f"exploring child {child.text}")
            if(child.dep_ not in EXCLUSIONS):
                self.algo2_sentence_dfs(child,context)
        return context

    def algo3_execute(self, text):
        log.debug(f"Processing text: {text}")
        doc = self.nlp(text)
        for sentence in doc.sents:
            log.debug(f"Executing {sentence=}")
            print(f"Executing {sentence=}")
            sentence_uuid = str(uuid.uuid4())
            phrase_triplets, dict_triplets = self.algo3_sentencer(sentence)
            log.debug(f"Executing {phrase_triplets=}, {dict_triplets=}")
            self.process_save_ners_tokens(sentence_uuid, sentence, [phrase_triplets, dict_triplets])
            
            sql_str = "select * from vw_sentences where sentence_uuid = ?"
            params = (sentence_uuid, )
            vw_text_df = pd.read_sql(sql_str, self.db, params=params)
            G_sent = self.algo3_create_graph(dict_triplets, vw_text_df)
            self.G = nx.compose(self.G, G_sent)

        nx.write_gexf(self.G, GEXF_PATH)
        # nx.write_graphml(self.G, GRAPHML_PATH)

    def algo3_sentencer(self, doc):
        DEP_SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
        POS_LINKS = ["AUX","ADP", "CCONJ", "PART"] # more research might be needed here. could do a mix of POS & DEP
        DEP_OBJECTS = ["pobj", "dative","oprd"]
        DEP_ATTRIBUTES = ["attr"]
        DEP_ACTIVITIES = ["dobj"]
        POS_NOUN = "NOUN"
        DEP_MODIFIERS = ["compound", "npadvmod"]
        ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod", "ccomp", "complm", "hmod", "infmod", "xcomp", "rcmod", "poss"," possessive"] # might need to add this to DEP_X_NOUNS
        DEP_X_NOUNS = DEP_OBJECTS + DEP_ATTRIBUTES + DEP_ACTIVITIES + DEP_MODIFIERS

        def reset_phrases():
            return "","-","","",""

        subject_phrase, link_phrase, object_phrase, attribute_phrase, activity_phrase = reset_phrases()
        phrase_triplets = []
        dict_triplets = []
        current_phrase = ""
        source_link = ""
        last_subject = ""
        doc = self.algo3_pre_process_sentence(doc)
        for token in doc:
            if token.dep_ == "punct":
                continue

            current_phrase = f"{current_phrase} {token.text}"
            log.debug(f"1. {current_phrase=}")

            if token.dep_ in DEP_SUBJECTS:
                
                if last_subject != "":
                    phrase_triplet = [last_subject, link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
                    dict_triplet = [{LABEL: last_subject, CLASSIFICATION:ENTITY} ,
                            {LABEL: link_phrase, TYPE: LINK},
                            {LABEL: current_phrase.lstrip(), CLASSIFICATION:ENTITY}]
                    log.debug(f"1.1 {dict_triplet=}")
                    phrase_triplets.append(phrase_triplet)
                    dict_triplets.append(dict_triplet)
                    link_phrase = "-"

                subject_phrase = current_phrase.lstrip()
                last_subject = subject_phrase
                source_link = subject_phrase
                current_phrase = ""
                log.debug(f"2. {subject_phrase=}")
            
            if token.pos_ in POS_LINKS:
                link_phrase = current_phrase.lstrip()
                current_phrase = ""
                log.debug(f"3. {link_phrase=}")

            if token.dep_ in DEP_OBJECTS:
                object_phrase = current_phrase.lstrip()
                current_phrase = ""
                log.debug(f"4.1 {object_phrase=}")

            if token.pos_ == POS_NOUN and token.dep_ not in DEP_X_NOUNS:
                object_phrase = current_phrase.lstrip()
                current_phrase = ""
                log.debug(f"4.2 {object_phrase=}")                

            if token.dep_ in DEP_ATTRIBUTES:
                attribute_phrase = current_phrase.lstrip()
                current_phrase = ""
                log.debug(f"5. {attribute_phrase=}")
            
            if token.dep_ in DEP_ACTIVITIES:
                activity_phrase = current_phrase.lstrip()
                current_phrase = ""
                log.debug(f"6. {activity_phrase=}")

            log.debug(f"7. {source_link=}, {subject_phrase=}, {link_phrase=}, {object_phrase=}, {attribute_phrase=}, {activity_phrase=}, {current_phrase=}, {last_subject=}")

            if len(subject_phrase) > 0 and len(attribute_phrase) > 0:
                phrase_triplet = [subject_phrase, link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
                dict_triplet = [{LABEL:subject_phrase, CLASSIFICATION:ENTITY}, {LABEL: link_phrase, TYPE: LINK}, {LABEL:attribute_phrase, CLASSIFICATION:ATTRIBUTE}]
                log.debug(f"8. {dict_triplet=}")
                phrase_triplets.append(phrase_triplet)
                dict_triplets.append(dict_triplet)
                
                source_link = attribute_phrase
                subject_phrase, link_phrase, object_phrase, attribute_phrase, activity_phrase = reset_phrases()

            if len(source_link) > 0 and (len(object_phrase) > 0 or len(activity_phrase)) :
                phrase_triplet = [source_link, link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
                right = {}
                if len(object_phrase) > 0:
                    right = {LABEL:object_phrase, CLASSIFICATION:ENTITY}
                if len(activity_phrase):
                    right = {LABEL:activity_phrase, CLASSIFICATION:ACTIVITY}
                dict_triplet = [{LABEL:source_link, CLASSIFICATION:ENTITY},{LABEL: link_phrase, TYPE: LINK}, right]
                log.debug(f"9. {dict_triplet=}")
                phrase_triplets.append(phrase_triplet)
                dict_triplets.append(dict_triplet)
                subject_phrase, link_phrase, object_phrase, attribute_phrase, activity_phrase = reset_phrases()        

            log.debug(f"10. {source_link=}, {subject_phrase=}, {link_phrase=}, {object_phrase=}, {attribute_phrase=}, {activity_phrase=}, {current_phrase=}, {last_subject=}")
        
        if len(object_phrase) > 0 or len(attribute_phrase) > 0 or len(activity_phrase) > 0 or len(current_phrase)>0:
            phrase_triplet = [source_link, link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
            right = {}
            if len(attribute_phrase) > 0:
                right = {LABEL:attribute_phrase, CLASSIFICATION:ATTRIBUTE}
            if len(object_phrase)>0:
                right = {LABEL:object_phrase, CLASSIFICATION:ENTITY}
            if len(activity_phrase)>0:
                right = {LABEL:activity_phrase, CLASSIFICATION:ACTIVITY}
            if len(current_phrase) > 0:
                right = {LABEL:current_phrase, CLASSIFICATION:ATTRIBUTE}
            dict_triplet = [{LABEL:source_link, CLASSIFICATION:ENTITY},{LABEL: link_phrase, TYPE: LINK}, right]
            log.debug(f"11. {dict_triplet=}")
            phrase_triplets.append(phrase_triplet)
            dict_triplets.append(dict_triplet)            
            
        return phrase_triplets, dict_triplets

    # Very tricky function to transform <x's y> or <s' y> as <y of x> or <y of s>
    # To me this is coded as more of a hack rather than with any elegance. But gets the job done
    # The challenge is finding the cases, and then swapping each case; which changes the order
    # and hence needs to be reparsed. So one will find things like the no_of_cases is just used
    # for number of loops and then doc is reparsed to find new indexes of the case, etc.
    def algo3_pre_process_sentence(self, doc):
        
        POS_NOUN_CHUNK_MODIFIERS = ["NOUN","PROPN","ADJ", "ADV"]
        DEP_APOSTROPHE = 'case'
        
        log.debug(f"pre_processing: {[[token.i, token.text, token.pos_, token.dep_] for token in doc]}")
        words = [token.text for token in doc]
        no_of_cases = len([token.i for token in doc if token.dep_ == DEP_APOSTROPHE])

        if no_of_cases == 0:
            return doc

        for case_i in range(no_of_cases):
            sentence = ' '.join(words)
            sentence = sentence.replace(" '", "'")
            doc = self.nlp(sentence)
            cases = [token.i for token in doc if token.dep_=="case"]
            case = cases[0] # Since we are popping each case at the end of the loop, the cases[0] always addresses next case
            
            # Find the noun chunk BEFORE case
            noun_chunk_1 = []
            for token in reversed(doc[:case]):
                if token.pos_ in POS_NOUN_CHUNK_MODIFIERS:
                    noun_chunk_1.append(token.i)
                else:
                    break  
    
            # Find the noun chunk AFTER case
            noun_chunk_2 = []
            for token in doc[case+1:] :
                noun_chunk_2.append(token.i)
                if token.pos_ in ["NOUN", "PROPN"]:
                    break

            pop_from = noun_chunk_1[-1]
            insert_at = noun_chunk_2[-1]+1
            words.insert(insert_at, "of")
            for j in noun_chunk_1:
                words.insert(insert_at, words.pop(pop_from))
            words.pop(pop_from)
            log.debug(f"words at end of loop: {words}")
        
        sentence = ' '.join([word for word in words if word not in ("'s", "'")])
        return self.nlp(sentence)

    def get_node(self, G, search_text):
        node_list = []
        for node_label in list(G.nodes()):
            if search_text in node_label:
                node_list.append(node_label)
        return node_list

    def algo3_create_graph(self, dict_triplets, text_df):
        G = nx.MultiDiGraph()
        for triplet in dict_triplets:
            nodes = [(triplet[0][LABEL], {CLASSIFICATION:triplet[0][CLASSIFICATION]}),(triplet[2][LABEL], {CLASSIFICATION:triplet[2][CLASSIFICATION]})]
            link = (triplet[0][LABEL], triplet[2][LABEL], {LINK_LABEL:triplet[1][LABEL]})
            G.add_nodes_from(nodes)
            G.add_edges_from([link])

            # G.add_nodes([(head[LABEL], {CLASSIFICATION:head[CLASSIFICATION]})])
            # G.add_node(tail[LABEL], Classification=tail[CLASSIFICATION])
            # G.add_edge(head[LABEL], tail[LABEL], Label=link[LABEL])

        ner_list = []
        for index, row in text_df.iterrows():
            if row["TYPE"] == NER:
                ner_list.append(row["item"])

        for index, row in text_df.iterrows():            
            if row["TYPE"] == NER:
                ner_item = row["item"]
                node_list = self.get_node(G, ner_item)
                for node_label in node_list:
                    node = (ner_item, {CLASSIFICATION:ENTITY, TYPE: row["NER_type"] })
                    G.add_nodes_from([node])
                    # G.add_node(ner_item, Classification=ENTITY, Type=row["NER_type"]) # If same as original node entity, it will just add attribute
                    self.add_meta_nodes(G, row, ["wdInstance","wikiDataClass","dbPediaType"])

                    if node_label != ner_item :
                        link = (node_label, ner_item, {TYPE:NER})
                        G.add_edges_from([link])
                        # G.add_edge(node_label, ner_item, Type=NER)

            if row["TYPE"] == "TOKEN" and row["token_pos"] in ["PROPN","NOUN"] and row["item"] not in ner_list:
                noun_item = row["item"]
                node_list = self.get_node(G, noun_item)
                for node_label in node_list:
                    node = (noun_item, {CLASSIFICATION:ENTITY, TYPE:NOUN}) # If same as original node entity, it will just add attribute
                    G.add_nodes_from([node])
                    # G.add_node(noun_item, Classification=ENTITY, Type="Noun") # If same as original node entity, it will just add attribute
                    self.add_meta_nodes(G, row, ["wdInstance","wikiDataClass","dbPediaType"])

                    if (row["token_pos"] == "NOUN") and str(row["list_conceptNetType"]) != "None":
                        self.add_meta_nodes(G, row, ["conceptNetType"])

                    if node_label != noun_item :
                        link = (node_label, noun_item, {TYPE:NOUN})
                        G.add_edges_from([link])
                        # G.add_edge(node_label, noun_item, Type="Noun")
        
        return G


    def get_processor(self, algo):
        if (algo=="algo1"):
            return self.algo1_execute
        elif (algo=="algo2"):
            return self.algo2_execute
        elif (algo=="algo3"):
            return self.algo3_execute
        else:
            raise Error(f"Can't find suggested algo {algo}")

    '''
    Thinking of the following flow:

    ToDo:
    - For really large sentences the spo algo doesn't work well. Think of pre-processing to break it down to smaller sentences - I think algo3 solves it
    - Make this a CLI app using Typer
    - Graph database
        - decision neo4j or arangodb?? - https://medium.com/neo4j/nxneo4j-networkx-api-for-neo4j-a-new-chapter-9fc65ddab222
    
    '''

def main():
    if len(sys.argv) < 3:
        print("Please provide params:\n1) algo_name(algo1|algo2)\n2)interaction_type (inline|file)\n3) if param2 is file - provide full filepath")
        exit(0)
    tp = TextProcessor()

    if sys.argv[2] == "file":
        if len(sys.argv) != 4:
            print("Please provide full filename as 3rd parameter")
            exit(0)
        
        with open(sys.argv[3]) as fp: 
            lines = fp.readlines() 
            for line in lines:
                processor = tp.get_processor(sys.argv[1])
                log.info(f"Processing line: {line}")
                processor(line.rstrip())
        log.info("Done")
    else:
        text=input("Para: ")
        while(text!="/stop"):
            # tp.algo1_execute(text)
            processor = tp.get_processor(sys.argv[1])
            processor(text)
            print("Done...")
            text = input("Para: ")

if __name__=="__main__":
    main()