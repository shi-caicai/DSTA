import time

import requests

'''
查询qid对应的label
'''
def get_label(args,property_id):
    url = "https://www.wikidata.org/w/api.php"
    language = 'en'
    if args.dataset == 'weibo':
        language = 'zh'
    params = {
        "action": "wbgetentities",
        "ids": property_id,
        "props": "labels",
        "languages": language,  # 指定语言为英语
        "format": "json"
    }
    retries=2
    while retries>0:
        try:
            requests.packages.urllib3.disable_warnings()
            proxies = {"http": "http://172.30.118.161:7890", "https": "http://172.30.118.161:7890"}
            response = requests.get(url, params=params, verify=False,proxies=proxies)
            data = response.json()
            if "entities" in data and property_id in data["entities"]:
                label = data["entities"][property_id]["labels"][language]["value"]
                return label
            else:
                return None
        except Exception:
            time.sleep(2)
            retries-=1
    return None