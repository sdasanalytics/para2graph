#----------------------------#
# Author: Surjit Das
# Email: surjitdas@gmail.com
# Program: artmind
#----------------------------#

import constants as C
from loguru import logger as log
import conceptnet_lite as cn_l
import urllib
from string import punctuation
import json
import sys
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
from datetime import datetime
import sqlite3

cn_l.connect(C.CONCEPTNET_LOCAL_DB)

class Explorer:
    def __init__(self):
        self.db = sqlite3.connect(C.SQL_EXT_KB_DB)
    
    def get_conceptnet_data(self, text):
        conceptnet_list = []
        try:
            for e in cn_l.edges_for(cn_l.Label.get(text=text, language='en').concepts, same_language=True):
                if(e.relation.name in ['is_a'] and e.start.text == text):
                    conceptnet_list.append(e.end.text)
        except:
            conceptnet_list.append("UNKNOWN")
        return conceptnet_list

    def wikifier(self, text, lang="en", threshold=0.8):
        """Function that fetches entity linking results from wikifier.com API"""
        # Prepare the URL.
        data = urllib.parse.urlencode([
            ("text", text), ("lang", lang),
            ("userKey", C.WIKIFIER_USER_KEY) # this is my userkey
            ,
            ("pageRankSqThreshold", "%g" %
            threshold), ("applyPageRankSqThreshold", "false"), # This flag is important - true seems to filter out a lot
            ("nTopDfValuesToIgnore", "100"), ("nWordsToIgnoreFromList", "100"),
            ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
            ("support", "true"), ("ranges", "false"), ("minLinkFrequency", "2"),
            ("includeCosines", "false"), ("maxMentionEntropy", "3")
        ])
    
        # Call the Wikifier and read the response.
        req = urllib.request.Request(C.WIKIFIER_URL, data=data.encode("utf8"), method="POST")
        with urllib.request.urlopen(req, timeout=60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))
        
        log.debug(data)
        log.debug(response)
        """
            The response is of the following JSON format
            {'annotations': 
                [
                    {
                        'title': 'Diwali', 'url': 'http://en.wikipedia.org/wiki/Diwali', ...
                        'wikiDataClasses': [
                                            {'itemId': 'Q132241', 'enLabel': 'festival'},
                                            {'itemId': 'Q1445650', 'enLabel': 'holiday'},
                                            {'itemId': 'Q15275719', 'enLabel': 'recurring event'}, ...
                                            ],
                        'dbPediaTypes': ['Holiday'],
                        'dbPediaIri': 'http://dbpedia.org/resource/Apple', ...
                    }
                ],
                'spaces': ['', ' ', ''],
                'words': ['Diwali'],...
            }
        """
        
        # results = filter_wikifier_response(response)
        wk_dict = {C.COL_WIKIDATACLASS:[],C.COL_DBPEDIA:[]}
        for record in response["annotations"]:
            # print(record)
            wdClassList = []
            wikiDataClasses = record.get('wikiDataClasses',[])
            for item in wikiDataClasses:
                wdClassList.append(item['enLabel'])
            dbpTypeList = record['dbPediaTypes']

            wk_dict[C.COL_WIKIDATACLASS].extend(wdClassList)
            wk_dict[C.COL_DBPEDIA].extend(dbpTypeList)

        return wk_dict

    # For wikidata, defining a generic sparql calling function
    def get_sparql_results(self, query):
        user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
        # TODO adjust user agent; see https://w.wiki/CX6
        sparql = SPARQLWrapper(C.WIKIDATA_SPARQL_ENDPOINT_URL, agent=user_agent)
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
        response = requests.get(C.WIKIDATA_API_ENDPOINT_URL, params=params)
        response_json = response.json()
        log.debug(f"get_wikidata::response_json: {response_json}")
        entity_ids = []
        for entity in response_json['search']:
            entity_ids.append(entity['id'])
        
        records_dict = {}
        # Put an empty list if wd does not find the entity
        if len(entity_ids)==0:
            records_dict[C.COL_WDINSTANCE] = ["wd_UNKNOWN"]
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
        
        records_dict[C.COL_WDINSTANCE] = list(records_dict.keys())
        
        log.debug(records_dict)
        return records_dict

    def get_ext_kb_info(self, text):
        '''
        Get's all the external kb information from each of the external data sources and returns them as a dict
        It first checks if the information about the token is available in the database, if so returns from db
        Otherwise makes the API/web calls, saves it to db for later use and then returns
        Thus this is a fully encapsulated function
        '''
        sql_str = f"select * from {C.TAB_EXT_KBS} where {C.COL_ITEM}=?"
        params = (text,)
        df = pd.read_sql(sql_str, self.db, params=params)
        df = df.fillna("[]")
        if len(df) != 0:
            return {C.COL_WIKIDATACLASS:df[C.COL_WIKIDATACLASS][0], C.COL_WDINSTANCE:df[C.COL_WDINSTANCE][0],
                        C.COL_DBPEDIA:df[C.COL_DBPEDIA][0], C.COL_CONCEPTNET:df[C.COL_CONCEPTNET][0]}
        else:
            wk_dict = self.wikifier(text)
            wikidataclass = str(wk_dict[C.COL_WIKIDATACLASS])
            dbpedia = str(wk_dict[C.COL_DBPEDIA])
            wdinstance = str(self.get_wikidata(text)[C.COL_WDINSTANCE])
            conceptnet = str(self.get_conceptnet_data(text.lower()))
            ts = datetime.now()
            
            cols = [C.COL_ITEM,C.COL_WIKIDATACLASS, C.COL_DBPEDIA, C.COL_WDINSTANCE,C.COL_CONCEPTNET, C.COL_TS]
            
            df = pd.DataFrame([[text, wikidataclass,dbpedia,wdinstance,conceptnet,ts]],columns=cols)
            df.to_sql(C.TAB_EXT_KBS, self.db, if_exists="append", index=False)
            return {C.COL_WIKIDATACLASS:wikidataclass, C.COL_WDINSTANCE:dbpedia, C.COL_DBPEDIA:wdinstance, C.COL_CONCEPTNET:conceptnet}

def test(text):
    exp = Explorer()
    x = exp.get_ext_kb_info("Southern")
    print(x[C.COL_WIKIDATACLASS])
    print(x[C.COL_WDINSTANCE])
    print(x[C.COL_DBPEDIA])
    print(x[C.COL_CONCEPTNET])

if __name__=="__main__":
    test("mathematics")

