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

SUBJECT = "SUBJECT"
PREDICATE = "PREDICATE"
OBJECT = "OBJECT"
NOUN = "NOUN"
VERB_PHRASE = "ACTIVITIES"
NER = "NER"

WIKIDATA_API_ENDPOINT_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_ENDPOINT_URL = "https://query.wikidata.org/sparql"
CONCEPTNET_API_ENDPOINT_URL = "http://api.conceptnet.io/c/en/"

nlp = spacy.load("en_core_web_trf")
cn_l.connect("/Volumes/Surjit_SSD_1/tech/conceptnet.db")
log.basicConfig(level=log.INFO, filename='/Users/surjitdas/Downloads/log_nlu_processor.log', filemode='w', format='%(message)s')

# pos = Parts of Speech - hence pos_it
def pos_it(doc):
    log.debug("|token.text| token.dep_| token.pos_| token.head.text|token.lemma_|")
    for token in doc:
        log.debug(f"|{token.text:<12}| {token.dep_:<10}| {token.pos_:<10}| {token.head.text:12}|{token.lemma_:12}")

# My own function invented to create the best chunks out of the sentences
def get_spacy_data(doc):
    subject = []
    subject_words = []
    predicate = []
    object_ = []
    noun = []
    verb_phrase = []
    
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
        log.debug(f"Processing token: {token.text} ({token.dep_} - {token.pos_})")
        if 'dobj' == token.dep_:
            verb_phrase.append(f"{token.head.text} {token.text}")
            log.debug(f"verb_phrase:{verb_phrase}")
        if 'NOUN' == token.pos_ or 'PROPN' == token.pos_:
            noun.append(token.lemma_)

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
        ent_list = ner_dict.get(ent.label_,[])
        ent_list.append(ent.text)
        ner_dict[ent.label_] = ent_list

    spacy_data = {
            SUBJECT:subject,
            PREDICATE:predicate,
            OBJECT:object_,
            NOUN: noun,
            VERB_PHRASE: verb_phrase,
            NER: ner_dict
            }
    
    log.debug(spacy_data)
    return spacy_data

# I don't intend to use wikifier, I think direct intefaces to wikidata & conceptnet are better suited for this SideProject
def wikifier(text, lang="en", threshold=0.8):
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
    url = "http://www.wikifier.org/annotate-article"
    # Call the Wikifier and read the response.
    req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
    with urllib.request.urlopen(req, timeout=60) as f:
        response = f.read()
        response = json.loads(response.decode("utf8"))
    
    log.debug(data)
    log.debug(response)
    # results = filter_wikifier_response(response)
    wikifier_dict_list = []
    for record in response["annotations"]:
        # print(record)
        wk_dict = {}
        wdClassList = []
        wikiDataClasses = record['wikiDataClasses']
        for item in wikiDataClasses:
            wdClassList.append(item['enLabel'])
        dbpTypeList = record['dbPediaTypes']
        wk_dict['wikiDataClasses'] = wdClassList
        wk_dict['DBPEDIATYPES'] = dbpTypeList
        wikifier_dict_list.append(wk_dict)

    return wikifier_dict_list

def filter_wikifier_response(response):
    ENTITY_TYPES = ["human", "person", "company", "enterprise", "business", "geographic region",
                "human settlement", "geographic entity", "territorial entity type", "organization"]
    results = list()
    for annotation in response["annotations"]:
        # Filter out desired entity classes
        if ('wikiDataClasses' in annotation) and (any([el['enLabel'] in ENTITY_TYPES for el in annotation['wikiDataClasses']])):
            # Specify entity label
            if any([el['enLabel'] in ["human", "person"] for el in annotation['wikiDataClasses']]):
                label = 'Person'
            elif any([el['enLabel'] in ["company", "enterprise", "business", "organization"] for el in annotation['wikiDataClasses']]):
                label = 'Organization'
            elif any([el['enLabel'] in ["geographic region", "human settlement", "geographic entity", "territorial entity type"] for el in annotation['wikiDataClasses']]):
                label = 'Location'
            else:
                label = None

            results.append({'title': annotation['title'], 'wikiId': annotation['wikiDataItemId'], 'label': label,
                            'characters': [(el['chFrom'], el['chTo']) for el in annotation['support']]})
    return results

# For wikidata, defining a generic sparql calling function
def get_sparql_results(query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(WIKIDATA_SPARQL_ENDPOINT_URL, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def get_wikidata(text, limit=1):

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
    log.debug(response_json)
    entity_ids = []
    for entity in response_json['search']:
        entity_ids.append(entity['id'])
    
    records_dict = {}
    if len(entity_ids)==0:
        records_dict["text"] = "No data in wikidata"
        return records_dict

    entity_id = entity_ids[0] # First entity_id. If there are none, this will throw an error
    query =(f"SELECT ?instance_of ?instance_ofLabel "
            f"?subclass_of ?subclass_ofLabel "
            f"WHERE {{wd:{entity_id} "
            f"wdt:P31 ?instance_of . "
            f"?instance_of wdt:P279+ ?subclass_of . "
            f"SERVICE wikibase:label {{ bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. }}}}")
    log.debug(query)
    results = get_sparql_results(query)
    log.debug(results)
    records = (results["results"]["bindings"])
    
    for record in records:
        column1 = record['instance_ofLabel']['value']
        column2 = record['subclass_ofLabel']['value']
        col2_list = records_dict.get(column1,[])
        col2_list.append(column2)
        records_dict[column1] = col2_list
    
    records_dict["INSTANCE_OF"] = list(records_dict.keys())

    return records_dict

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

def get_conceptnet_data(text):
    conceptnet_list = []
    try:
        for e in cn_l.edges_for(cn_l.Label.get(text=text, language='en').concepts, same_language=True):
            if(e.relation.name in ['is_a'] and e.start.text == text):
                conceptnet_list.append(e.end.text)
    except:
        conceptnet_list.append("UNKNOWN")
    return conceptnet_list

def process_nouns(nouns):
    nouns_list = []
    for noun in nouns:
        log.info(f"Processing: {noun}...")
        attrs = {}
        attrs['wikidata'] = get_wikidata(noun)
        attrs['wikifier'] = wikifier(noun)
        attrs['conceptnet'] = get_conceptnet_data(noun)
        nouns_list.append([noun, attrs])
    return nouns_list

def process_verb_phrases(verb_phrases):
    verb_phrases_list = []
    for verb_phrase in verb_phrases:
        log.info(f"Processing: {verb_phrase}...")
        attrs = {'dummy':'value'}
        verb_phrases_list.append([verb_phrase, attrs])
    return verb_phrases_list

def main():
    text=input("Para: ")
    while(text!="/stop"):
        log.debug(text)
        doc = nlp(text)
        pos_it(doc)
        spacy_data = get_spacy_data(doc)
        log.info(spacy_data)
        G = nx.Graph()
        
        nouns = process_nouns(spacy_data[NOUN])
        log.debug(nouns)
        G.add_nodes_from(nouns, type="Entity")
        
        activities = process_verb_phrases(spacy_data[VERB_PHRASE])
        log.debug(activities)
        G.add_nodes_from(activities, type="Activity")
        for node in G.nodes(data=True):
            log.info(node)
        plot_graph(G)

        text = input("Para: ")

def test():
    # for e in cn_l.edges_for(cn_l.Label.get(text='lamps', language='en').concepts, same_language=True):
    #         print(e)
    x = get_conceptnet_data('lamp')
    print(x)

if __name__=="__main__":
    main()