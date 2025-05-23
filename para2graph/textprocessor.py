#----------------------------#
# Author: Surjit Das
# Email: surjitdas@gmail.com
# Program: artmind
#----------------------------#

from types import TracebackType

from numpy.lib.twodim_base import tri
from py2neo.matching import NE
import constants as C
import spacy
from loguru import logger as log
import uuid
from p2g_dataclasses import PhraseNode, PhraseEdge, SentenceGraph, SentenceTable, NERNode, NounNode, PhraseInfoEdge, KBNode, AdjNode, VerbNode
from external_kbs import Explorer
from wordnet_explorer import WordNet_Explorer
import sqlite3
import py2neo as p2n
import ast

class TextProcessor:
    """
    The TextProcessor contains the main execution logic for Para2Graph
    """
    def __init__(self, mode="truncate"):
        self.nlp = spacy.load(C.SPACY_MODEL)
        self.db = sqlite3.connect(C.SQL_LOCAL_DB)
        self.kbs = Explorer()
        self.G_n4j = p2n.Graph(C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
        if mode == "truncate":
            self.G_n4j.delete_all()        
    
    def execute(self, text):
        """
        The main driver loop for TextProcessor

        :param text: the text to be processed. This can be full paragraph as well, because this function breaks it down into sentences and then processes by each sentence
        """
        doc = self.nlp(text)
        for sentence in doc.sents:
            log.debug(f"Executing {sentence=}")

            '''
            Create a unique identifier using uuid. This uuid is associated for each sentence. This uuid is used for:
            - storing the sentence tokens in database
            - storing uuid as a property for each phrase (this will help in creating different nodes per sentence even if words are same across sentences)
            '''
            sentence_uuid = str(uuid.uuid4())
            
            '''
            Pre process the sentence. If multiple pre-processing needs to be done, add here...
            '''
            sentence = self.preprocess_sentence_for_apostrophe(sentence)

            '''
            Persist the sentence tokens in db
            '''
            s_t = SentenceTable(self.db)
            ners, nouns, adjs, verbs = s_t.persist(sentence_uuid, sentence)
            deduped_nouns = self.dedup_nouns_from_ners(nouns, ners)
            
            '''
            Break sentences into phrases and get PhraseEdges
            '''
            ph_3plets = self.sentencer(sentence_uuid, sentence)

            '''
            Get edges that are Phrase->Noun | NER | Adjective | Verb
            '''
            ner_pos_3plets = self.construct_phrase_3plets(ners, deduped_nouns, adjs, verbs, ph_3plets)

            '''
            Get edges that are Noun|NER->KB_Info 
            '''
            kb_3plets = self.constuct_kb_3plets(ners, deduped_nouns, adjs, verbs)

            '''
            Save the outcomes to persistent graph
            '''
            s_g = SentenceGraph(self.G_n4j, sentence_uuid)
            s_g.save(ph_3plets, ner_pos_3plets, kb_3plets)

    def dedup_nouns_from_ners(self, nouns, ners):
        '''
        Removes the nouns that are in the NERs
        '''
        if len(ners) == 0:
            return nouns

        ner_words, ner_types = zip(*ners)
        set_ner_words = set((" ").join(ner_words).split())
        return list(set(nouns) - set_ner_words)

    def construct_phrase_3plets(self, ners, deduped_nouns, adjs, verbs, ph_3plets):
        '''
        This function takes the ners, pos (nouns, adjs, verbs) and the phrase triplets and creates PhraseInfoEdges
        '''
        # de-duped phrase node triplets
        phrases = []
        for triplet in ph_3plets:
            # print(f"{triplet.head=}, {triplet.phrase=}, {triplet.tail=}")
            if triplet.head not in phrases:
                phrases.append(triplet.head)
            if triplet.tail not in phrases:
                phrases.append(triplet.tail)

        ner_pos_3plets = []
        for phrase in phrases:
            for ner in ners:
                if ner[0] in phrase.phrase:
                    ner_pos_3plets.append(PhraseInfoEdge(phrase, NERNode(ner[0], ner[1])))
            for noun in deduped_nouns:
                if noun.lower() in phrase.phrase.lower():
                    ner_pos_3plets.append(PhraseInfoEdge(phrase, NounNode(noun)))
            for adj in adjs:
                if adj.lower() in phrase.phrase.lower():
                    ner_pos_3plets.append(PhraseInfoEdge(phrase, AdjNode(adj)))
            for verb in verbs:
                if verb.lower() in phrase.phrase.lower():
                    ner_pos_3plets.append(PhraseInfoEdge(phrase, VerbNode(verb)))      

        log.debug(f"{ner_pos_3plets=}")              
        return ner_pos_3plets
          
    def constuct_kb_3plets(self, ners, deduped_nouns, adjs, verbs):
        '''
        Takes the NER | Pos creates NER | Pos -> KB_Info (for all 5 KBs) nodes + edges
        '''
        kb_3plets = []

        for ner in ners:
            kb_info = self.kbs.get_ext_kb_info(ner[0])
            kb_3plets = self.add_meta_nodes(NERNode(ner[0],ner[1]), kb_info, kb_3plets, [C.WIKIDATA_CLASS, C.DBPEDIA, C.WDINSTANCE, C.CONCEPTNET])

        for noun in deduped_nouns:
            # kb_info = self.kbs.get_ext_kb_info(noun)
            # kb_3plets = self.add_meta_nodes(NounNode(noun), kb_info, kb_3plets, [C.WIKIDATA_CLASS, C.DBPEDIA, C.WDINSTANCE, C.CONCEPTNET])
            kb_3plets = self.add_wordnet_nodes(NounNode(noun), kb_3plets, noun)

        for adj in adjs:
            kb_3plets = self.add_wordnet_nodes(AdjNode(adj), kb_3plets, adj)

        for verb in verbs:
            kb_3plets = self.add_wordnet_nodes(VerbNode(verb), kb_3plets, verb)            

        log.debug(f"{kb_3plets=}")   
        return kb_3plets

    def add_wordnet_nodes(self, head, kb_3plets, pos):
        '''
        Utility function used by constuct_kb_3plets to create the KB_Info node + edges for WordNet parent classes
        '''        
        wn_e = WordNet_Explorer(pos.lower())
        parents = wn_e.get_parent_classes()
        if len(parents)>0:
            kb_3plets.append(PhraseInfoEdge(head,KBNode(parents[0], C.WORDNET)))        
            i = 0
            while True:
                try:
                    kb_3plets.append(PhraseInfoEdge(KBNode(parents[i], C.WORDNET),KBNode(parents[i+1], C.WORDNET)))
                    i = i + 1
                except IndexError:
                    break
        return kb_3plets

    def add_meta_nodes(self, head, kb_info, kb_3plets, sources, max_nodes=C.MAX_KB_NODES):
        '''
        Utility function used by constuct_kb_3plets to create the KB_Info node + edges
        It iterates by each type of source
        '''
        for source in sources:
            # The kb_info coming from the external_kbs.Explorer returns a list as a string. That's how it is stored in the db too.
            # The ast.literal_eval function converts that string into a list
            label_list = ast.literal_eval(kb_info[f"list_{source}"])
            for label in label_list[:max_nodes]:
                if label not in ['Wikimedia disambiguation page', 'MediaWiki main-namespace page', 'list', 'class', 
                            'word-sense disambiguation', 'Wikimedia internal item', 'MediaWiki page', 'MediaWiki help page','Wikimedia non-main namespace',
                            'wd_UNKNOWN', 'UNKNOWN']:
                    tail = KBNode(label, source)
                    kb_3plets.append(PhraseInfoEdge(head,tail))
        return kb_3plets
        
    def preprocess_sentence_for_apostrophe(self, doc):
        """
        Very tricky function to transform <x's y> or <s' y> as <y of x> or <y of s>   
        To me this is coded as more of a hack rather than with any elegance. But gets the job done
        The challenge is finding the cases, and then swapping each case; which changes the order
        and hence needs to be reparsed. So one will find things like the no_of_cases is just used
        for number of loops and then doc is reparsed to find new indexes of the case, etc.        
        """
        '''
        The constants below control a lot of the logic and are part of the "researched" hacks :-)
        '''
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

    def sentencer(self, sentence_uuid, doc):
        '''
        Breaks down the sentence into phrases and saves those in the Graph.
        First it breaks it down into phrases and linked phrases - subject, attribute, object, linked phrase
        :param sentence_uuid: the unique id for the sentence under which this loop is running
        :param doc: the nlp doc form of the sentence to be processed
        --------------------------------------------------------------
        The logic of the sentencer is quite interesting and can be described as follows:
        Each sentence is made up of phrases. There are different types of phrases. These phrase types are determined by Stop words and their corresponding types.
        | Phrase Type | Stop Word Type |
        | ---         | ---            |
        | Subject     | DEP_SUBJECTS   |
        | Object      | DEP_OBJECTS + escaping NOUNS |
        | Attribute   | DEP_ATTRIBUTES |
        | Activities  | DEP_ACTIVITIES |
        | Link        | POS_LINKS      |
        Part I: Phrase boundary detection
        - While going through each token in the sentence, a current phrase is constructed till a Stop word is found
        - When a Stop word is found, a phrase is completed and put in a phrase Q or bucket
        Part II: Triplet boundary detection
        - The next step is to find a triplet. This is done by checking the status of phrase Qs or buckets. When specific conditions are met, a phrase triplet is constructed
        - To construct phrase triplet, the context of subject Q and object Qs are important to link the phrase nodes
        - The phrase triplets are then added to a triplet Q
        And the above is repeated till end of sentence
        '''

        '''
        Stop word Constants: The constants below have a huge impact on the logic. They have been chosen with care. But more scenarios may bring in changes
        '''
        DEP_SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
        POS_LINKS = ["AUX","ADP", "CCONJ", "PART"] # more research might be needed here. could do a mix of POS & DEP
        DEP_OBJECTS = ["pobj", "dative","oprd"]
        DEP_ATTRIBUTES = ["attr"]
        DEP_ACTIVITIES = ["dobj"]
        DEP_MODIFIERS = ["compound", "npadvmod"]
        ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod", "ccomp", "complm", "hmod", "infmod", "xcomp", "rcmod", "poss"," possessive"] # Not used might need to add this to DEP_X_NOUNS
        DEP_X_NOUNS = DEP_OBJECTS + DEP_ATTRIBUTES + DEP_ACTIVITIES + DEP_MODIFIERS + DEP_SUBJECTS
        DEP_PUNCT = "punct"

        def reset_phrases():
            '''
            Inner function to reset subject_phrase, link_phrase, object_phrase, attribute_phrase, activity_phrase 
            '''
            return "","-","","",""

        '''
        Phrase buckets are instantiated for the 5 types of phrases
        '''
        subject_phrase, link_phrase, object_phrase, attribute_phrase, activity_phrase = reset_phrases()
        
        '''
        The context Qs for subject and object
        '''
        last_subject = ""
        object_list = []
        source_link = []
        
        '''
        The phrase triplet Qs
        '''
        phrase_triplets = []
        ph_3plets = []

        current_phrase = ""
        
        log.debug("|token.text| token.dep_| token.pos_| token.head.text|token.lemma_|")
        for token in doc:
            log.debug(f"|{token.text:<12}| {token.dep_:<10}| {token.pos_:<10}| {token.head.text:12}|{token.lemma_:12}")

        for token in doc:            
            if token.dep_ == DEP_PUNCT: # Ignore punctuations
                continue

            '''
            Part I: Phrase boundary detection
            '''
            current_phrase = f"{current_phrase} {token.text}"
            log.debug(f"1. {token.text=}, {current_phrase=}")

            # If token in Subject Stop Word, complete Subject phrase, add to subject context and reset current phrase
            if token.dep_ in DEP_SUBJECTS:

                # Part II: A special case of triplet boundary detection for 2nd subject. Might be able to refactor this code to be more elegant
                if last_subject != "":
                    phrase_triplet = [last_subject, link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
                    ph_3plet = PhraseEdge(PhraseNode(sentence_uuid, last_subject, C.SUBJECT), "-",
                                        PhraseNode(sentence_uuid, current_phrase.lstrip(), C.SUBJECT),
                                        sentence_uuid)
                    log.debug(f"2.1 {ph_3plet=}")
                    phrase_triplets.append(phrase_triplet)
                    ph_3plets.append(ph_3plet)
                    link_phrase = "-"
                    object_list=[]

                subject_phrase = current_phrase.lstrip()
                last_subject = subject_phrase
                source_link = [subject_phrase, C.SUBJECT]
                current_phrase = ""
                log.debug(f"2.1 {subject_phrase=}")
            
            # If token in Link Stop Word, complete Link phrase and reset current phrase
            if token.pos_ in POS_LINKS:
                link_phrase = current_phrase.lstrip()
                current_phrase = ""
                log.debug(f"3. {link_phrase=}")

            # If token in Object Stop Word, complete Object phrase, add to Object context and reset current phrase
            if token.dep_ in DEP_OBJECTS:
                object_phrase = current_phrase.lstrip()
                object_list.append(object_phrase)
                current_phrase = ""
                log.debug(f"4.1 {object_phrase=}")

            # If token is an ESCAPING Noun, complete Object phrase, add to Object context and reset current phrase
            if token.pos_ == C.POS_NOUN and token.dep_ not in DEP_X_NOUNS:
                object_phrase = current_phrase.lstrip()
                object_list.append(object_phrase)
                current_phrase = ""
                log.debug(f"4.2 {object_phrase=}")                

            # If token in Attribute Stop Word, complete Attribute phrase and reset current phrase
            if token.dep_ in DEP_ATTRIBUTES:
                attribute_phrase = current_phrase.lstrip()
                current_phrase = ""
                log.debug(f"5. {attribute_phrase=}")
            
            # If token in Activities Stop Word, complete Activities phrase and reset current phrase
            if token.dep_ in DEP_ACTIVITIES:
                activity_phrase = current_phrase.lstrip()
                current_phrase = ""
                log.debug(f"6. {activity_phrase=}")

            log.debug(f"7. {source_link=}, {subject_phrase=}, {link_phrase=}, {object_phrase=}, {attribute_phrase=}, {activity_phrase=}, {current_phrase=}, {last_subject=}, {object_list=}")

            '''
            Part II: Triplet boundary detection
            '''
            # If there is a Subject phrase as well as an Attribute phrase, then complete phrase triplet and add to triplet Q
            # Triplet boundary condition : Subject-[-]->Attribute
            if len(subject_phrase) > 0 and len(attribute_phrase) > 0:
                phrase_triplet = [subject_phrase, link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
                ph_3plet = PhraseEdge(PhraseNode(sentence_uuid, subject_phrase, C.SUBJECT), link_phrase,
                                    PhraseNode(sentence_uuid, attribute_phrase, C.ATTRIBUTE),
                                    sentence_uuid)
                log.debug(f"8. {ph_3plet=}")
                phrase_triplets.append(phrase_triplet)
                ph_3plets.append(ph_3plet)
                source_link = [attribute_phrase, C.ATTRIBUTE]
                subject_phrase, link_phrase, object_phrase, attribute_phrase, activity_phrase = reset_phrases()

            # If there is a Subject phrase as well as an Attribute phrase, then complete phrase triplet and add to triplet Q
            # Triplet boundary condition : Subject|Attribute|Object-[-]->Object|Activity
            if len(source_link) > 0 and (len(object_phrase) > 0 or len(activity_phrase)>0) :
                if len(object_list)>1:
                    source_link = [object_list[-2], C.OBJECT]      
                phrase_triplet = [source_link[0], link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
                right = {}
                if len(object_phrase) > 0:
                    right = {C.NODE_TEXT:object_phrase, C.CLASSIFICATION:C.OBJECT}
                if len(activity_phrase):
                    right = {C.NODE_TEXT:activity_phrase, C.CLASSIFICATION:C.ACTIVITY}
                ph_3plet = PhraseEdge(PhraseNode(sentence_uuid, source_link[0], source_link[1]), link_phrase,
                    PhraseNode(sentence_uuid, right[C.NODE_TEXT], right[C.CLASSIFICATION]),
                    sentence_uuid)
                
                log.debug(f"9. {ph_3plet=}, {link_phrase=}")
                phrase_triplets.append(phrase_triplet)
                ph_3plets.append(ph_3plet)
                subject_phrase, link_phrase, object_phrase, attribute_phrase, activity_phrase = reset_phrases()        

            log.debug(f"10. {source_link=}, {subject_phrase=}, {link_phrase=}, {object_phrase=}, {attribute_phrase=}, {activity_phrase=}, {current_phrase=}, {last_subject=}, {object_list=}")
        
        if len(object_phrase) > 0 or len(attribute_phrase) > 0 or len(activity_phrase) > 0 or len(current_phrase)>0:
            if len(source_link)==0:
                if subject_phrase != '':
                    source_link = [subject_phrase, C.SUBJECT]
                elif len(object_list)>1:
                    source_link = [object_list[-2], C.OBJECT] 
                else:
                    source_link = ["-","-"]

            phrase_triplet = [source_link[0], link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
            right = {}
            if len(attribute_phrase) > 0:
                right = {C.NODE_TEXT:attribute_phrase, C.CLASSIFICATION:C.ATTRIBUTE}
            if len(object_phrase)>0:
                right = {C.NODE_TEXT:object_phrase, C.CLASSIFICATION:C.OBJECT}
            if len(activity_phrase)>0:
                right = {C.NODE_TEXT:activity_phrase, C.CLASSIFICATION:C.ACTIVITY}
            if len(current_phrase) > 0:
                right = {C.NODE_TEXT:current_phrase, C.CLASSIFICATION:C.ATTRIBUTE}
            ph_3plet = PhraseEdge(PhraseNode(sentence_uuid, source_link[0], source_link[1]), link_phrase,
                PhraseNode(sentence_uuid, right[C.NODE_TEXT], right[C.CLASSIFICATION]),
                sentence_uuid)                

            log.debug(f"11. {ph_3plet=}")
            phrase_triplets.append(phrase_triplet)
            ph_3plets.append(ph_3plet)
        
        log.debug(f"12. {ph_3plets=}")
            
        return ph_3plets
