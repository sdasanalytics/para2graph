#----------------------------#
# Author: Surjit Das
# Email: surjitdas@gmail.com
# Program: artmind
#----------------------------#

import constants as C
import spacy
from loguru import logger as log
import uuid
from p2g_dataclasses import PhraseNode, PhraseEdge, SentenceGraph
import sqlite3
import py2neo as p2n

class TextProcessor:
    """
    The TextProcessor contains the main execution logic for Para2Graph
    """
    def __init__(self, mode="truncate"):
        self.nlp = spacy.load(C.SPACY_MODEL)
        self.db = sqlite3.connect(C.SQL_LOCAL_DB)
        # self.G = nx.MultiDiGraph() <-- Change
        # self.kbs = external_kbs.Explorer() <-- Change
        self.G_n4j = p2n.Graph(C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
        if mode == "append":
        #     self.G = nx.read_gexf(C.GEXF_PATH)
            ...
        else:
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
            Break sentences into phrases
            '''
            self.sentencer(sentence_uuid, sentence)
            

            # phrase_triplets, dict_triplets = self.algo3_sentencer(sentence_uuid, sentence)
            # self.process_save_ners_tokens(sentence_uuid, sentence, [phrase_triplets, dict_triplets])
            
            # sql_str = f"select * from {C.VW_SENTENCES} where {C.COL_SENT_UUID} = ?"
            # params = (sentence_uuid, )
            # vw_text_df = pd.read_sql(sql_str, self.db, params=params)
            # G_sent = self.algo3_create_graph(dict_triplets, vw_text_df)
            # self.G = nx.compose(self.G, G_sent)

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
        """
        Breaks down the sentence into phrases and saves those in the Graph.
        First it breaks it down into phrases and linked phrases - subject, attribute, object, linked phrase
        :param sentence_uuid: the unique id for the sentence under which this loop is running
        :param doc: the nlp doc form of the sentence to be processed
        """

        '''
        The constants below have a huge impact on the logic. Have been chosen with care.
        But more scenarios can bring in changes
        '''
        DEP_SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
        POS_LINKS = ["AUX","ADP", "CCONJ", "PART"] # more research might be needed here. could do a mix of POS & DEP
        DEP_OBJECTS = ["pobj", "dative","oprd"]
        DEP_ATTRIBUTES = ["attr"]
        DEP_ACTIVITIES = ["dobj"]
        DEP_MODIFIERS = ["compound", "npadvmod"]
        ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod", "ccomp", "complm", "hmod", "infmod", "xcomp", "rcmod", "poss"," possessive"] # might need to add this to DEP_X_NOUNS
        DEP_X_NOUNS = DEP_OBJECTS + DEP_ATTRIBUTES + DEP_ACTIVITIES + DEP_MODIFIERS + DEP_SUBJECTS
        DEP_PUNCT = "punct"

        def reset_phrases():
            '''
            Inner function to reset subject_phrase, link_phrase, object_phrase, attribute_phrase, activity_phrase 
            '''
            return "","-","","",""

        subject_phrase, link_phrase, object_phrase, attribute_phrase, activity_phrase = reset_phrases()
        phrase_triplets = []
        # dict_triplets = []
        ph_3lets = []
        current_phrase = ""
        source_link = []
        last_subject = ""
        object_list = []
        
        log.debug("|token.text| token.dep_| token.pos_| token.head.text|token.lemma_|")
        for token in doc:
            log.debug(f"|{token.text:<12}| {token.dep_:<10}| {token.pos_:<10}| {token.head.text:12}|{token.lemma_:12}")

        '''
        Loop through each token and construct phrases
        This loop is broken into 3 main parts:
        - Part 1. This creates a span of words into a phrase. It ends the phrase when it finds a stop word of a particular type
        - Part 2. It creates a graph triplet (?)
        - Part 3. ??
        '''
        for token in doc:            
            if token.dep_ == DEP_PUNCT: # Ignore punctuations
                continue

            '''
            Part 1:
            1. Add the current token to the phrase
            2.1. In the case there are 2 ore more subjects in the sentence, then it links them together
            2.2. Continues on to find subject phrase 
            '''
            current_phrase = f"{current_phrase} {token.text}"
            log.debug(f"1. {token.text=}, {current_phrase=}")

            if token.dep_ in DEP_SUBJECTS:
                
                if last_subject != "":
                    phrase_triplet = [last_subject, link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
                    # dict_triplet = [{C.NODE_TEXT: last_subject, C.CLASSIFICATION:C.SUBJECT} ,
                    #         {C.NODE_TEXT: link_phrase, C.CLASSIFICATION: C.LINK, C.PHRASE_TYPE:C.LINK},
                    #         {C.NODE_TEXT: current_phrase.lstrip(), C.CLASSIFICATION:C.SUBJECT}]
                    ph_3let = PhraseEdge(PhraseNode(sentence_uuid, last_subject, C.SUBJECT), "-",
                                        PhraseNode(sentence_uuid, current_phrase.lstrip(), C.SUBJECT),
                                        sentence_uuid)

                    log.debug(f"2.1 {ph_3let=}")
                    phrase_triplets.append(phrase_triplet)
                    # dict_triplets.append(dict_triplet)
                    ph_3lets.append(ph_3let)
                    link_phrase = "-"
                    object_list=[] # Added

                subject_phrase = current_phrase.lstrip()
                last_subject = subject_phrase
                source_link = [subject_phrase, C.SUBJECT]
                current_phrase = ""
                log.debug(f"2.1 {subject_phrase=}")
            
            if token.pos_ in POS_LINKS:
                link_phrase = current_phrase.lstrip()
                current_phrase = ""
                log.debug(f"3. {link_phrase=}")

            if token.dep_ in DEP_OBJECTS:
                object_phrase = current_phrase.lstrip()
                object_list.append(object_phrase)
                current_phrase = ""
                log.debug(f"4.1 {object_phrase=}")

            if token.pos_ == C.POS_NOUN and token.dep_ not in DEP_X_NOUNS:
                object_phrase = current_phrase.lstrip()
                object_list.append(object_phrase)
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

            log.debug(f"7. {source_link=}, {subject_phrase=}, {link_phrase=}, {object_phrase=}, {attribute_phrase=}, {activity_phrase=}, {current_phrase=}, {last_subject=}, {object_list=}")

            if len(subject_phrase) > 0 and len(attribute_phrase) > 0:
                phrase_triplet = [subject_phrase, link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
                # dict_triplet = [{C.NODE_TEXT:subject_phrase, C.CLASSIFICATION:C.SUBJECT}, 
                #                 {C.NODE_TEXT: link_phrase, C.CLASSIFICATION: C.LINK}, 
                #                 {C.NODE_TEXT:attribute_phrase, C.CLASSIFICATION:C.ATTRIBUTE}]
                ph_3let = PhraseEdge(PhraseNode(sentence_uuid, subject_phrase, C.SUBJECT), link_phrase,
                                    PhraseNode(sentence_uuid, attribute_phrase, C.ATTRIBUTE),
                                    sentence_uuid)

                log.debug(f"8. {ph_3let=}")
                phrase_triplets.append(phrase_triplet)
                # dict_triplets.append(dict_triplet)
                ph_3lets.append(ph_3let)
                
                source_link = [attribute_phrase, C.ATTRIBUTE]
                subject_phrase, link_phrase, object_phrase, attribute_phrase, activity_phrase = reset_phrases()

            if len(source_link) > 0 and (len(object_phrase) > 0 or len(activity_phrase)>0) :
                if len(object_list)>1: # Added
                    source_link = [object_list[-2], C.OBJECT]  # Added               
                phrase_triplet = [source_link[0], link_phrase, attribute_phrase, object_phrase, activity_phrase, current_phrase]
                right = {}
                if len(object_phrase) > 0:
                    right = {C.NODE_TEXT:object_phrase, C.CLASSIFICATION:C.OBJECT}
                if len(activity_phrase):
                    right = {C.NODE_TEXT:activity_phrase, C.CLASSIFICATION:C.ACTIVITY}
                # dict_triplet = [{C.NODE_TEXT:source_link[0], C.CLASSIFICATION:source_link[1]},{C.NODE_TEXT: link_phrase, C.CLASSIFICATION: C.LINK}, right]
                ph_3let = PhraseEdge(PhraseNode(sentence_uuid, source_link[0], source_link[1]), link_phrase,
                    PhraseNode(sentence_uuid, right[C.NODE_TEXT], right[C.CLASSIFICATION]),
                    sentence_uuid)
                
                log.debug(f"9. {ph_3let=}, {link_phrase=}")
                phrase_triplets.append(phrase_triplet)
                # dict_triplets.append(dict_triplet)
                ph_3lets.append(ph_3let)
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
            # dict_triplet = [{C.NODE_TEXT:source_link[0], C.CLASSIFICATION:source_link[1]},{C.NODE_TEXT: link_phrase, C.CLASSIFICATION: C.LINK}, right]
            ph_3let = PhraseEdge(PhraseNode(sentence_uuid, source_link[0], source_link[1]), link_phrase,
                PhraseNode(sentence_uuid, right[C.NODE_TEXT], right[C.CLASSIFICATION]),
                sentence_uuid)                

            log.debug(f"11. {ph_3let=}")
            phrase_triplets.append(phrase_triplet)
            # dict_triplets.append(dict_triplet)
            ph_3lets.append(ph_3let)
        
        log.debug(f"12. {ph_3lets=}")
        s_g = SentenceGraph(self.G_n4j, sentence_uuid)
        s_g.save(ph_3lets)

        # for triplet in dict_triplets:
        #     for item in triplet:
        #         item[C.UUID] = sentence_uuid
            
        # return phrase_triplets, dict_triplets

    
            