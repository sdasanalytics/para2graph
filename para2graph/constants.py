#----------------------------#
# Author: Surjit Das
# Email: surjitdas@gmail.com
# Program: artmind
#----------------------------#

SQL_LOCAL_DB = "/Users/surjitdas/Downloads/nlu_processor/nlu_processor_v2.db"
GEXF_PATH = "/Users/surjitdas/Downloads/nlu_processor/nlu_processor.gexf"
LOG_PATH = '/Users/surjitdas/Downloads/nlu_processor/nlu_processor.log'

WIKIFIER_URL = "http://www.wikifier.org/annotate-article"
WIKIDATA_API_ENDPOINT_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_ENDPOINT_URL = "https://query.wikidata.org/sparql"
# CONCEPTNET_API_ENDPOINT_URL = "http://api.conceptnet.io/c/en/" not using the Web API, but directly the local database & API
CONCEPTNET_LOCAL_DB = "/Volumes/Surjit_SSD_1/tech/conceptnet.db"

NEO4J_USER = 'neo4j'
NEO4J_PASSWORD = "unonothing"
NEO4J_URI = "bolt://localhost:7687"

# SPACY_MODEL = "en_core_web_lg"
SPACY_MODEL = "en_core_web_trf"

TAB_SENTENCES = "sentences"
TAB_EXT_KBS = "external_kbs"
VW_SENTENCES = "vw_sentences"
COL_SENT_UUID = "sentence_uuid"
COL_TYPE = "TYPE"
COL_NER_TYPE = "NER_type"
COL_ITEM = "item"
COL_TOKEN_DEP = "token_dep"
COL_TOKEN_POS = "token_pos"
COL_TOKEN_HEAD_TEXT = "token_head_text"
COL_TOKEN_LEMMA = "token_lemma"
COL_COMP_NOUN = "compound_noun"
COL_VERB_PHRASE = "verb_phrase"
COL_WDINSTANCE = "list_wdInstance"
COL_WIKIDATACLASS = "list_wikiDataClass"
COL_DBPEDIA = "list_dbPediaType"
COL_CONCEPTNET = "list_conceptNetType"
COL_TS = "ts"

COLUMNS_KBS_SOURCES = [COL_WDINSTANCE, COL_WIKIDATACLASS, COL_DBPEDIA, COL_CONCEPTNET]
COLUMNS_TOKEN = [COL_TOKEN_DEP, COL_TOKEN_POS, COL_TOKEN_HEAD_TEXT, COL_TOKEN_LEMMA, COL_COMP_NOUN, COL_VERB_PHRASE]
COLUMNS_KBS = [COL_ITEM] + COLUMNS_KBS_SOURCES + [COL_TS]
COLUMNS_PARA = [COL_SENT_UUID, COL_TYPE, COL_NER_TYPE, COL_ITEM] + COLUMNS_TOKEN + [COL_TS]
COLUMNS_DF = [COL_SENT_UUID, COL_TYPE, COL_NER_TYPE, COL_ITEM] + COLUMNS_TOKEN + COLUMNS_KBS_SOURCES + [COL_TS]

COL_TYPE_VAL_PARSED = "PARSED"
COL_TYPE_VAL_TOKEN = "TOKEN"

NER = "NER"

POS_PROPER_NOUN = "PROPN"
POS_NOUN = "NOUN"
DEP_DOBJ = "dobj"
DEP_COMPOUND = "compound"

PREDICATE = "PREDICATE"
ENTITY = "Entity"
ENTITYTYPE = "Entity_Type"
PHRASE_TYPE = "Phrase_Type"
LINK = "Link"
SUBJECT = "Subject"
OBJECT = "Object"
ATTRIBUTE = "Attribute"
ACTIVITY = "Activity"
NODE_TEXT = "Node_Text"
LINK_LABEL = "Link_Label"
CLASSIFICATION = "classification"
SOURCE = "source"
NOUN = "Noun"
PHRASE = "Phrase"
PHRASE_LINK = "Phrase_Link"
N4J_NODE_NAME = "name"


WD_INSTANCE = "wdInstance"
WIKIDATA_CLASS = "wikiDataClass"
DBPEDIA_TYPE = "dbPediaType"
CONCEPTNET_TYPE = "conceptNetType"

UUID = "uuid"
