from sparql_query import sparal_query
from wikidata_api import get_label
import pandas as pd
from tqdm import tqdm
import json
import os

uris = {'http://www.wikidata.org/entity/': 'wd', 'http://www.wikidata.org/prop/direct/': 'wdt',
        'http://www.wikidata.org/prop/': 'p', 'http://www.wikidata.org/entity/statement/': 'wds',
        'http://www.wikidata.org/prop/statement/': 'ps', 'http://www.wikidata.org/prop/qualifier/': 'pq'}

qid_relation_dict = {}
statement_value={}

def load_qid_relation(args):
    file_path = f"{args.path}/fnd/qid_relation.txt"
    if args.dataset=='weibo':
        file_path=f"{args.path}/fnd/qid_relation_zh.txt"
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            first_index = line.strip().find(';')
            qid = line.strip()[0:first_index]
            relationLabel = line.strip()[first_index+1:]
            qid_relation_dict[qid] = relationLabel

def load_statement_value(args):
    file_path = f"{args.path}/fnd/statement_value.txt"
    if args.dataset=='weibo':
        file_path=f"{args.path}/fnd/statement_value_zh.txt"
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            first_index = line.strip().find(';')
            statement = line.strip()[0:first_index]
            value = line.strip()[first_index + 1:]
            # statement, value = line.strip().split(';')
            statement_value[statement] = value

def extract_relation_qid(args,string):
    # 找到最后一个斜杠的位置
    last_slash_index = string.rfind('/')
    # 如果找到斜杠
    if last_slash_index != -1:
        uri = string[:last_slash_index+1]
        qid = string[last_slash_index+1:]
        if uri in uris.keys():
            if qid in qid_relation_dict:
                relationLabel = qid_relation_dict[qid]
            else:
                relationLabel = get_label(args,qid)
                if relationLabel!=None:
                    file_path = f"{args.path}/fnd/qid_relation.txt"
                    if args.dataset == 'weibo':
                        file_path = f"{args.path}/fnd/qid_relation_zh.txt"
                    with open(file_path, 'a', encoding="utf-8") as file:
                        file.write(f"{qid};{relationLabel}\n")
                    qid_relation_dict[qid] = relationLabel
            return uris[uri]+':'+qid, relationLabel
        elif 'schema' in uri or 'altLabel' in qid or 'rdfs' in qid:
            return qid, qid
        else:
            return None, None
    else:
        return string, string

# 通过返回label可以直接得到qid对应的实体名或属性名，但是长的标识符比如q80805-07970587-B118-4598-97D6-F6910CDBBB81无法直接得到
# 直接通过statement去查对应的label查不到，因为他是另一种很长的标识符比如q80805-07970587-B118-4598-97D6-F6910CDBBB81，通过?statement ps:{property[2:]} ?value.得到对应的短的qid也就是?value
def get_entity_property_statement_value(args, entity_qid, property, statement):
    # 构造 SPARQL 查询
    language = 'en'
    if args.dataset=='weibo':
        language='zh'
    query = f"""
        SELECT ?statement ?value ?valueLabel
        WHERE {{
          wd:{entity_qid} {property} ?statement.
          ?statement ps:{property[2:]} ?value.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],{language}". }}
        }}
        """
    results=sparal_query(query)
    for result in results:
        if result['statement']['value'] == statement:
            file_path = f"{args.path}/fnd/statement_value.txt"
            if args.dataset == 'weibo':
                file_path = f"{args.path}/fnd/statement_value_zh.txt"
            with open(file_path, 'a', encoding="utf-8") as file:
                file.write(f"{statement};{result['valueLabel']['value']}\n")
            statement_value[statement] = result['valueLabel']['value']
            return result['valueLabel']['value']
    return None

def step2(args, qid, entity_name):
    language = 'en'
    if args.dataset=='weibo':
        language='zh'
    query=f"""SELECT DISTINCT ?relationshipLabel ?neighbor ?neighborLabel
            WHERE {{
            wd:{qid} ?relationship ?neighbor.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],{language}". }}
            }}
            """
    '''
    results是个数组
    每个元素是
    {'relationshipLabel': {'type': 'literal', 'value': 'http://schema.org/version'}, 'neighborLabel': {'type': 'literal', 'value': '2147301039'}}
    '''
    results = sparal_query(query)
    if len(results)==0:
        return
    relation=[]
    oneHopNeighbor=[]
    oneHopNeighborLabel=[]
    for result in results:
        if (args.dataset=='weibo' and ('xml:lang' not in result['neighbor'] or result['neighbor']['xml:lang']=='zh')) or\
                ((args.dataset=='politifact' or args.dataset=='gossipcop')and ('xml:lang' not in result['neighbor'] or result['neighbor']['xml:lang']=='en')):
            relation.append(result['relationshipLabel']['value'])
            oneHopNeighbor.append(result['neighbor']['value'])
            oneHopNeighborLabel.append(result['neighborLabel']['value'])

    data={'relation':relation,'relationLabel':relation,'neighbor':oneHopNeighbor,'neighborLabel':oneHopNeighborLabel}
    df=pd.DataFrame(data)

    os.makedirs(f"{args.path}/fnd/wikidata_filteredData", exist_ok=True)
    os.makedirs(f"{args.path}/fnd/wikidata_filteredData_zh", exist_ok=True)
    filteredData_path = f"{args.path}/fnd/wikidata_filteredData/{qid}.csv"
    if args.dataset=='weibo':
        filteredData_path=f"{args.path}/fnd/wikidata_filteredData_zh/{qid}.csv"
    if not os.path.exists(path):
        # 处理relation，relationLabel列
        for _, row in df.iterrows():
            row['relation'], row['relationLabel'] = extract_relation_qid(args,row['relation'])
        # 删除'relationLabel'列为None的行
        df = df[df['relationLabel'].notna()]
        # 删除'relationLabel'列中值包含'ID'字符串的行
        df = df[~df['relationLabel'].str.contains('ID')]
        df.to_csv(filteredData_path, index=False)  # index=False 是为了不将索引写入 CSV

    df = pd.read_csv(filteredData_path)
    # 处理neighbor列
    for _,row in df.iterrows():
        if pd.isna(row['neighborLabel']):
            continue
        if '/' in row['neighborLabel']:
            index = row['neighbor'].rfind('/')
            if row['neighbor'][:index+1] in uris.keys():
                if row['neighbor'] in statement_value:
                    row['neighborLabel'] = statement_value[row['neighbor']]
                else:
                    row['neighborLabel']=get_entity_property_statement_value(args,qid,row['relation'],row['neighbor'])
            else:
                row['neighborLabel']=None
    # 删除'neighborLabel'列为None的行
    df = df[df['neighborLabel'].notna()]
    # 删除'neighborLabel'列中值包含'http'字符串的行
    df = df[~df['neighborLabel'].str.contains('http')]

    kg_triple=[]
    for _,row in df.iterrows():
        kg_triple.append((entity_name,row['relationLabel'],row['neighborLabel']))

    os.makedirs(f"{args.path}/fnd/wikidata_triplets_zh", exist_ok=True)
    os.makedirs(f"{args.path}/fnd/wikidata_triplets", exist_ok=True)
    output_file_path = f"{args.path}/fnd/wikidata_triplets/{qid}.json"
    if args.dataset=='weibo':
        output_file_path = f"{args.path}/fnd/wikidata_triplets_zh/{qid}.json"
    with open(output_file_path, 'w',encoding='utf-8') as f:
        json.dump(kg_triple, f)

if __name__ == '__main__':
    # 使用openie提取主谓宾三元组 然后把主语链接到维基百科页面 维基百科页面获取wikibase_item即qid
    step2('Q20145')