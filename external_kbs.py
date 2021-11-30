#----------------------------#
# Author: Surjit Das
# Email: surjitdas@gmail.com
# Program: artmind
#----------------------------#

from loguru import logger as log
import conceptnet_lite as cn_l
import urllib
from string import punctuation
import json
import sys
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

WIKIFIER_URL = "http://www.wikifier.org/annotate-article"
WIKIDATA_API_ENDPOINT_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_ENDPOINT_URL = "https://query.wikidata.org/sparql"
# CONCEPTNET_API_ENDPOINT_URL = "http://api.conceptnet.io/c/en/" not using the Web API, but directly the local database & API
CONCEPTNET_LOCAL_DB = "/Volumes/Surjit_SSD_1/tech/conceptnet.db"

cn_l.connect(CONCEPTNET_LOCAL_DB)

class Explorer:
    def __init__(self) -> None:
        pass
    
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
