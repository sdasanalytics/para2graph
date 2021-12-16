#----------------------------#
# Author: Surjit Das
# Email: surjitdas@gmail.com
# Program: artmind
#----------------------------#

from copy import Error
import constants as C
from loguru import logger as log
import spacy
import sys
from timefhuman import timefhuman as th
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import uuid
from datetime import datetime
import ast
import external_kbs
import py2neo as p2n
from tqdm import tqdm

log.remove() #removes default handlers
log.add(C.LOG_PATH, backtrace=True, diagnose=True, level="DEBUG")

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
ENTITYTYPE = "EntityType"
LINK = "Link"
ATTRIBUTE = "Attribute"
ACTIVITY = "Activity"
NODE_TEXT = "Node_Text"
LINK_LABEL = "Link_Label"
CLASSIFICATION = "classification"
SOURCE = "source"
NOUN = "Noun"
PHRASE = "Phrase"
N4J_NODE_NAME = "name"

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
        self.nlp = spacy.load(C.SPACY_MODEL)
        self.db = sqlite3.connect(C.SQL_LOCAL_DB)
        self.G = nx.MultiDiGraph()
        self.kbs = external_kbs.Explorer()

    def process_save_ners_tokens(self, sentence_uuid, sentence, parsed_context):
        text_df = pd.DataFrame(columns=C.COLUMNS_DF)
        parsed_context_row = {C.COL_SENT_UUID:sentence_uuid, C.COL_TYPE:C.COL_TYPE_VAL_PARSED, C.COL_ITEM: str(parsed_context), C.COL_TS : datetime.now()}
        text_df = text_df.append(parsed_context_row, ignore_index=True)
        for ent in sentence.ents:
            ner_row = {C.COL_SENT_UUID:sentence_uuid, C.COL_TYPE:C.NER, C.COL_ITEM:ent.text, C.COL_NER_TYPE:ent.label_, C.COL_TS : datetime.now()}
            sql_str = f"select * from {C.TAB_EXT_KBS} where {C.COL_ITEM}=?"
            params = (ent.text,)
            df = pd.read_sql(sql_str, self.db, params=params)
            if len(df) == 0:
                ner_row[C.COL_WDINSTANCE] = str(self.kbs.get_wikidata(ent.text)[C.COL_WDINSTANCE])
                wk_dict = self.kbs.wikifier(ent.text)
                ner_row[C.COL_WIKIDATACLASS] = str(wk_dict[C.COL_WIKIDATACLASS])
                ner_row[C.COL_DBPEDIA] = str(wk_dict[C.COL_DBPEDIA])
                
                kbs_dict = {C.COL_ITEM:[ent.text], 
                            C.COL_WDINSTANCE: [ner_row[C.COL_WDINSTANCE]], C.COL_WIKIDATACLASS: [ner_row[C.COL_WIKIDATACLASS]], C.COL_DBPEDIA: [ner_row[C.COL_DBPEDIA]],
                            C.COL_TS : [datetime.now()]}
                kbs_df = pd.DataFrame(kbs_dict)
                kbs_df.to_sql(C.TAB_EXT_KBS, self.db, index=False, if_exists="append")
            else:
                ner_row[C.COL_WDINSTANCE] = df[C.COL_WDINSTANCE][0]
                ner_row[C.COL_WIKIDATACLASS] = df[C.COL_WIKIDATACLASS][0]
                ner_row[C.COL_DBPEDIA] = df[C.COL_DBPEDIA][0]
            
            text_df = text_df.append(ner_row, ignore_index=True)

        text_df = self.process_tokens(sentence_uuid, sentence, text_df)
        
        # Write the text_df to db
        para_df = text_df[C.COLUMNS_PARA]
        para_df.to_sql(C.TAB_SENTENCES, self.db, index=False, if_exists="append")
        return text_df

    def add_meta_nodes(self, G, row, sources):
        for source in sources:
            label_list = ast.literal_eval(row[f"list_{source}"])
            for label in label_list[:3]:
                if label not in ['Wikimedia disambiguation page', 'MediaWiki main-namespace page', 'list', 'word-sense disambiguation', 'Wikimedia internal item', 'MediaWiki page', 'wd_UNKNOWN']:
                    head = "|"+row[C.COL_ITEM]+"|"
                    tail = (label, {CLASSIFICATION:ENTITYTYPE, SOURCE:source})
                    G.add_nodes_from([tail])
                    link = (head,label,{SOURCE:source})
                    G.add_edges_from([link])
                    log.debug(f"{head=}, {link=}, {tail=}")

    def save_graph(self, mode="append"):
        G_p2n = p2n.Graph(C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
        
        if mode == "append":
            self.G = nx.read_gexf(C.GEXF_PATH)
        else:
            G_p2n.delete_all()
        
        nx.write_gexf(self.G, C.GEXF_PATH)

        for node in tqdm(self.G.nodes(data=True), desc="Writing nodes to database:"):
            log.debug(f"{node=}")
            n4j_node_label = node[1].get(SOURCE, PHRASE)
            n4j_node_name = node[0]
            attrs = {N4J_NODE_NAME:n4j_node_name, CLASSIFICATION:node[1].get(CLASSIFICATION,"-")}
            log.debug(f"{n4j_node_label=}, {attrs=}")
            p2n_node = p2n.Node(n4j_node_label, **attrs)
            G_p2n.create(p2n_node)

        for edge in tqdm(self.G.edges(data=True), desc="Writing edges to database:"):
            log.debug(f"{edge=}")
            head_name = edge[0]
            head_n4j_node_label = self.G.nodes[head_name].get(SOURCE, PHRASE)
            head_n4j_node = G_p2n.nodes.match(head_n4j_node_label, name=head_name).first()

            tail_name = edge[1]
            tail_n4j_node_label = self.G.nodes[tail_name].get(SOURCE, PHRASE)
            tail_n4j_node = G_p2n.nodes.match(tail_n4j_node_label, name=tail_name).first()
            
            n4j_rel_name = edge[2].get(LINK_LABEL,"-")
            if n4j_rel_name == '':
                n4j_rel_name = "-"
            rel = p2n.Relationship.type(n4j_rel_name)
            n4j_rel_type = edge[2].get(SOURCE,PREDICATE)
            attrs = {SOURCE:n4j_rel_type}
            log.debug(f"{head_n4j_node=}, {tail_n4j_node=}, {attrs=}")
            link = rel(head_n4j_node, tail_n4j_node, **attrs)
            G_p2n.create(link)


    def algo1_execute(self, text):
        log.debug(f"Processing text: {text}")
        doc = self.nlp(text)
        for sentence in doc.sents:
            sentence_uuid = str(uuid.uuid4())
            spacy_data, text_df = self.algo1_process_sentence(sentence_uuid, sentence)
            self.algo1_create_graph(self.G, spacy_data, text_df)
        nx.write_gexf(self.G, C.GEXF_PATH)
        
        # for node in self.G.nodes(data=True):
        #     log.debug(node)
        # plot_graph(self.G)
        
        return

    def algo1_process_sentence(self, sentence_uuid, sentence):
        # Add subject, predicate, object & C.NER to the dataframe - as 1st row for the para/sentence
        spo_data = self.algo1_spacy_data(sentence)
        text_df = self.process_save_ners_tokens(sentence_uuid, sentence, spo_data)
        return spo_data, text_df

    def process_tokens(self, sentence_uuid, sentence, text_df):
        # Add tokens to the dataframe - 1 row per token as the 2nd row onwards for the para/sentence
        log.debug("|token.text| token.dep_| token.pos_| token.head.text|token.lemma_|")
        for token in sentence:
            log.debug(f"|{token.text:<12}| {token.dep_:<10}| {token.pos_:<10}| {token.head.text:12}|{token.lemma_:12}")
            sql_str = f"select * from {C.TAB_EXT_KBS} where {C.COL_ITEM}=?"
            params = (token.text,)
            df = pd.read_sql(sql_str, self.db, params=params)
            row = {C.COL_SENT_UUID:sentence_uuid, C.COL_TYPE:C.COL_TYPE_VAL_TOKEN, C.COL_ITEM:token.text, 
                    C.COL_TOKEN_DEP:token.dep_, C.COL_TOKEN_POS:token.pos_, C.COL_TOKEN_HEAD_TEXT:token.head.text, C.COL_TOKEN_LEMMA:token.lemma_}
            
            # Processing nouns
            if(token.pos_ in [C.POS_PROPER_NOUN,C.POS_NOUN]):
                if len(df) == 0:
                    row[C.COL_WDINSTANCE] = str(self.kbs.get_wikidata(token.text)[C.COL_WDINSTANCE])
                    
                    wk_dict = self.kbs.wikifier(token)
                    row[C.COL_WIKIDATACLASS] = str(wk_dict[C.COL_WIKIDATACLASS])
                    row[C.COL_DBPEDIA] = str(wk_dict[C.COL_DBPEDIA])
                    kbs_dict = {C.COL_ITEM:[token.text], 
                            C.COL_WDINSTANCE: [row[C.COL_WDINSTANCE]], C.COL_WIKIDATACLASS: [row[C.COL_WIKIDATACLASS]], C.COL_DBPEDIA: [row[C.COL_DBPEDIA]],
                            C.COL_TS : [datetime.now()]}
                    if(token.pos_ == C.POS_NOUN):
                        row[C.COL_CONCEPTNET] = str(self.kbs.get_conceptnet_data(token.lemma_.lower()))
                        kbs_dict[C.COL_CONCEPTNET] = [row[C.COL_CONCEPTNET]]

                    kbs_df = pd.DataFrame(kbs_dict)
                    kbs_df.to_sql(C.TAB_EXT_KBS, self.db, index=False, if_exists="append")
                else:
                    row[C.COL_WDINSTANCE] = df[C.COL_WDINSTANCE][0]
                    row[C.COL_WIKIDATACLASS] = df[C.COL_WIKIDATACLASS][0]
                    row[C.COL_DBPEDIA] = df[C.COL_DBPEDIA][0]
                    row[C.COL_CONCEPTNET] = df[C.COL_DBPEDIA][0]
                    
            # Processing compound nouns
            if(token.dep_ == C.DEP_COMPOUND):
                row[C.COL_COMP_NOUN] = f"{token.text} {token.head.text}"
            
            # Processing verb phrases
            if (token.dep_ == C.DEP_DOBJ) :
                row[C.COL_VERB_PHRASE] = f"{token.head.text} {token.text}"
            
            row[C.COL_TS] = datetime.now()
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
            
            if row[C.COL_TYPE] == C.NER:
                attrs = {CLASSIFICATION:ENTITY, SOURCE:row[C.COL_NER_TYPE]}
                G.add_node(row[C.COL_ITEM], **attrs)
                
                self.add_meta_nodes(G, row, ["wdInstance","wikiDataClass","dbPediaType"])

            if str(row[C.COL_COMP_NOUN]) != 'nan':
                pass # This is a ToDo ... Decide whether compound nouns need to be handled & whether this is the right place for this code

            if (row[C.COL_TYPE] == C.COL_TYPE_VAL_TOKEN and row[C.COL_TOKEN_POS] in ["PROPN","NOUN"] and (row[C.COL_ITEM] not in spacy_data["C.NER"])):
                attrs = {CLASSIFICATION:ENTITY}
                G.add_node(row[C.COL_ITEM], **attrs)
                
                self.add_meta_nodes(G, row, ["wdInstance","wikiDataClass","dbPediaType"])

                if (row[C.COL_TOKEN_POS] == "NOUN") and str(row[C.COL_CONCEPTNET]) != 'nan':
                    self.add_meta_nodes(G, row, ["conceptNetType"])
                
            if(row[C.COL_TOKEN_POS] == "PRON"):
                attrs = {CLASSIFICATION:ENTITY, SOURCE:"EntityPointer"}
                G.add_node(row[C.COL_ITEM], **attrs)

            if str(row[C.COL_VERB_PHRASE]) != 'nan':
                attrs = {CLASSIFICATION:ACTIVITY}
                G.add_node(row[C.COL_VERB_PHRASE], **attrs)
    
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
                C.NER: ner_dict
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
            sql_str = f"select * from {C.VW_SENTENCES} where {C.COL_SENT_UUID} = ?"
            params = (sentence_uuid, )
            vw_text_df = pd.read_sql(sql_str, self.db, params=params)

    def algo2_process_sentence(self, sentence_uuid, sentence):
        # Add subject, predicate, object & C.NER to the dataframe - as 1st row for the para/sentence
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
            sentence_uuid = str(uuid.uuid4())
            phrase_triplets, dict_triplets = self.algo3_sentencer(sentence)
            log.debug(f"Executing {phrase_triplets=}, {dict_triplets=}")
            self.process_save_ners_tokens(sentence_uuid, sentence, [phrase_triplets, dict_triplets])
            
            sql_str = f"select * from {C.VW_SENTENCES} where {C.COL_SENT_UUID} = ?"
            params = (sentence_uuid, )
            vw_text_df = pd.read_sql(sql_str, self.db, params=params)
            G_sent = self.algo3_create_graph(dict_triplets, vw_text_df)
            self.G = nx.compose(self.G, G_sent)

    def algo3_sentencer(self, doc):
        '''
        This algorithm should be fine tuned further for links
        Right now all Objects are linked to Subject or Attr of Subjects. 
        TODO: But would be good to have Object chaining links till next Subject is found
        '''
        DEP_SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
        POS_LINKS = ["AUX","ADP", "CCONJ", "PART"] # more research might be needed here. could do a mix of POS & DEP
        DEP_OBJECTS = ["pobj", "dative","oprd"]
        DEP_ATTRIBUTES = ["attr"]
        DEP_ACTIVITIES = ["dobj"]
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
        log.debug("|token.text| token.dep_| token.pos_| token.head.text|token.lemma_|")
        for token in doc:
            log.debug(f"|{token.text:<12}| {token.dep_:<10}| {token.pos_:<10}| {token.head.text:12}|{token.lemma_:12}")

        for token in doc:
            if token.dep_ == "punct":
                continue

            current_phrase = f"{current_phrase} {token.text}"
            log.debug(f"1. {current_phrase=}")

            if token.dep_ in DEP_SUBJECTS:
                
                if last_subject != "":
                    phrase_triplet = [last_subject, link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
                    dict_triplet = [{NODE_TEXT: last_subject, CLASSIFICATION:ENTITY} ,
                            {NODE_TEXT: link_phrase, SOURCE: LINK},
                            {NODE_TEXT: current_phrase.lstrip(), CLASSIFICATION:ENTITY}]
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

            if token.pos_ == C.POS_NOUN and token.dep_ not in DEP_X_NOUNS:
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
                dict_triplet = [{NODE_TEXT:subject_phrase, CLASSIFICATION:ENTITY}, {NODE_TEXT: link_phrase, SOURCE: LINK}, {NODE_TEXT:attribute_phrase, CLASSIFICATION:ATTRIBUTE}]
                log.debug(f"8. {dict_triplet=}")
                phrase_triplets.append(phrase_triplet)
                dict_triplets.append(dict_triplet)
                
                source_link = attribute_phrase
                subject_phrase, link_phrase, object_phrase, attribute_phrase, activity_phrase = reset_phrases()

            if len(source_link) > 0 and (len(object_phrase) > 0 or len(activity_phrase)) :
                phrase_triplet = [source_link, link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
                right = {}
                if len(object_phrase) > 0:
                    right = {NODE_TEXT:object_phrase, CLASSIFICATION:ENTITY}
                if len(activity_phrase):
                    right = {NODE_TEXT:activity_phrase, CLASSIFICATION:ACTIVITY}
                dict_triplet = [{NODE_TEXT:source_link, CLASSIFICATION:ENTITY},{NODE_TEXT: link_phrase, SOURCE: LINK}, right]
                log.debug(f"9. {dict_triplet=}")
                phrase_triplets.append(phrase_triplet)
                dict_triplets.append(dict_triplet)
                subject_phrase, link_phrase, object_phrase, attribute_phrase, activity_phrase = reset_phrases()        

            log.debug(f"10. {source_link=}, {subject_phrase=}, {link_phrase=}, {object_phrase=}, {attribute_phrase=}, {activity_phrase=}, {current_phrase=}, {last_subject=}")
        
        if len(object_phrase) > 0 or len(attribute_phrase) > 0 or len(activity_phrase) > 0 or len(current_phrase)>0:
            phrase_triplet = [source_link, link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
            right = {}
            if len(attribute_phrase) > 0:
                right = {NODE_TEXT:attribute_phrase, CLASSIFICATION:ATTRIBUTE}
            if len(object_phrase)>0:
                right = {NODE_TEXT:object_phrase, CLASSIFICATION:ENTITY}
            if len(activity_phrase)>0:
                right = {NODE_TEXT:activity_phrase, CLASSIFICATION:ACTIVITY}
            if len(current_phrase) > 0:
                right = {NODE_TEXT:current_phrase, CLASSIFICATION:ATTRIBUTE}
            dict_triplet = [{NODE_TEXT:source_link, CLASSIFICATION:ENTITY},{NODE_TEXT: link_phrase, SOURCE: LINK}, right]
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
        
        POS_NOUN_CHUNK_MODIFIERS = ["NOUN","PROPN","ADJ", "ADV", "NUM"]
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
            log.debug(f"{cases=}")
            case = cases[0] # Since we are popping each case at the end of the loop, the cases[0] always addresses next case
            
            # Find the noun chunk BEFORE case
            noun_chunk_1 = []
            for token in reversed(doc[:case]):
                if token.pos_ in POS_NOUN_CHUNK_MODIFIERS:
                    noun_chunk_1.append(token.i)
                else:
                    break  
            log.debug(f"{noun_chunk_1=}")
            # Find the noun chunk AFTER case
            noun_chunk_2 = []
            for token in doc[case+1:] :
                noun_chunk_2.append(token.i)
                if token.pos_ in [C.POS_PROPER_NOUN, C.POS_NOUN]:
                    break
            log.debug(f"{noun_chunk_2=}")
            
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

    def is_within(self, text, lst):
        found = 0
        for item in lst:
            if text in item:
                found = 1
        return found

    def algo3_create_graph(self, dict_triplets, text_df):
        G = nx.MultiDiGraph()
        for triplet in dict_triplets:
            nodes = [(triplet[0][NODE_TEXT], {CLASSIFICATION:triplet[0][CLASSIFICATION]}),(triplet[2][NODE_TEXT], {CLASSIFICATION:triplet[2][CLASSIFICATION]})]
            link = (triplet[0][NODE_TEXT], triplet[2][NODE_TEXT], {LINK_LABEL:triplet[1][NODE_TEXT]})
            log.debug(f"{nodes=}, {link=}")
            G.add_nodes_from(nodes)
            G.add_edges_from([link])

        ner_list = []
        for index, row in text_df.iterrows():
            if row[C.COL_TYPE] == C.NER:
                ner_list.append(row[C.COL_ITEM])
        
        log.debug(f"{ner_list=}")

        for index, row in text_df.iterrows():            
            if row[C.COL_TYPE] == C.NER:
                ner_item = row[C.COL_ITEM]
                node_list = self.get_node(G, ner_item)
                log.debug(f"{node_list=}")
                for node_label in node_list:
                    # node = (ner_item, {CLASSIFICATION:ENTITY, SOURCE: row[C.COL_NER_TYPE] })
                    node = ("|"+ner_item+"|", {CLASSIFICATION:ENTITY, SOURCE: row[C.COL_NER_TYPE] })
                    G.add_nodes_from([node])
                    self.add_meta_nodes(G, row, ["wdInstance","wikiDataClass","dbPediaType"])
                    log.debug(f"{node_label=}, {ner_item=}")
                    
                    # if node_label != ner_item :
                    #     link = (node_label, ner_item, {SOURCE:C.NER})
                    #     # link = (node_label, "|"+ner_item+"|", {SOURCE:C.NER})
                    #     G.add_edges_from([link])

                    # link = (node_label, ner_item, {SOURCE:C.NER})
                    link = (node_label, "|"+ner_item+"|", {SOURCE:C.NER})
                    G.add_edges_from([link])

            

            if row[C.COL_TYPE] == C.COL_TYPE_VAL_TOKEN and row[C.COL_TOKEN_POS] in [C.POS_PROPER_NOUN, C.POS_NOUN]:
                noun_item = row[C.COL_ITEM]
                log.debug(f"{not self.is_within(noun_item, ner_list)=} for {noun_item=}")
                if not self.is_within(noun_item, ner_list):
                    node_list = self.get_node(G, noun_item)
                    log.debug(f"{node_list=}, {noun_item=}")
                    for node_label in node_list:
                        node = ("|"+noun_item+"|", {CLASSIFICATION:ENTITY, SOURCE:NOUN}) # If same as original node entity, it will just add attribute
                        G.add_nodes_from([node])
                        self.add_meta_nodes(G, row, ["wdInstance","wikiDataClass","dbPediaType"])

                        if (row[C.COL_TOKEN_POS] == C.POS_NOUN) and str(row[C.COL_CONCEPTNET]) != "None":
                            self.add_meta_nodes(G, row, ["conceptNetType"])

                        # if node_label != noun_item :
                        #     # link = (node_label, noun_item, {SOURCE:NOUN})
                        #     link = (node_label, "|"+noun_item+"|", {SOURCE:NOUN})
                        #     G.add_edges_from([link])

                        # link = (node_label, noun_item, {SOURCE:NOUN})
                        link = (node_label, "|"+noun_item+"|", {SOURCE:NOUN})
                        G.add_edges_from([link])

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

@log.catch
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
            processor = tp.get_processor(sys.argv[1])
            lines = fp.readlines() 
            for line in tqdm(lines, desc="Processing sentences"):
                log.info(f"Processing line: {line}")
                processor(line.strip())
            tp.save_graph(mode="overwrite")
        log.info("Done")
    else:
        text=input("Para: ")
        processor = tp.get_processor(sys.argv[1])
        while(text!="/stop"):
            processor(text)
            print("Done...")
            text = input("Para: ")
        tp.save_graph(mode="overwrite")

if __name__=="__main__":
    main()