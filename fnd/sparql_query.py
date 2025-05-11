#coding=utf-8
import requests
import time

def sparal_query(query):
    endpoint_url = "https://query.wikidata.org/sparql"
    headers = {'User-Agent': 'MyBot'}
    payload = {
        'query': query,
        'format': 'json'
    }
    retries = 3
    while retries > 0:
        try:
            proxies= {"http":"http://172.30.118.161:7890","https":"http://172.30.118.161:7890"}
            requests.packages.urllib3.disable_warnings()
            r = requests.get(endpoint_url, params=payload, headers=headers, verify=False,proxies=proxies)
            results = r.json()
            return results["results"]["bindings"]
        except Exception:
            time.sleep(2)
            retries -= 1
    return []

if __name__ == '__main__':
    property = "p:P27"
    statement = {'type': 'uri', 'value': 'http://www.wikidata.org/entity/statement/q80805-07970587-B118-4598-97D6-F6910CDBBB81'}
    r = sparal_query("""SELECT DISTINCT ?relationshipLabel ?neighbor ?neighborLabel
             WHERE {
             wd:Q69 ?relationship ?neighbor.
             SERVICE wikibase:label { bd:serviceParam wikibase:language "zh". }
             }
                    """)
    print(r)