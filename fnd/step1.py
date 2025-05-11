import re
import os
import torch
import pandas as pd
import json
from tqdm import tqdm
from sparql_query import sparal_query
from step2 import step2,load_qid_relation,load_statement_value
# 从yayinere文件里面导入类yayinerre
from YaYiNerRe import YaYiNerRe
from transformers import BertTokenizer,BertModel
from torch.nn.functional import cosine_similarity
from torch_geometric.data.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding
import numpy as np
from sklearn.model_selection import train_test_split
import networkx as nx
import argparse


print("gpu是否可用：", torch.cuda.is_available())

def parse_args():
    parser = argparse.ArgumentParser(description="step1")
    parser.add_argument("--dataset", type=str, default='politifact')
    parser.add_argument("--path", type=str, default='/data2/data/csh/G-Retriever-main')
    # onlykg,onlynews,all
    parser.add_argument("--graph_name", type=str, default='all')
    parser.add_argument("--max_length", type=int, default=3800)
    args = parser.parse_args()
    return args

def find_word_positions(sentence, phrase):
    words = sentence.split(" ")
    start = -1
    end = -1

    for i in range(len(words)):
        current_phrase = " ".join(words[i:])  # 从当前单词开始的子串
        if current_phrase.startswith(phrase):
            start = i
            end = start + len(phrase.split())
            break

    return start, end

def find_sentence_with_entity(text, entity):
    # 分句，以句号、问号、感叹号作为分隔符
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\。|\.|\?|\？|\!|\！|\;|\；|\n)\s', text)

    # 查找包含目标实体的句子
    for sentence in sentences:
        # 使用正则表达式查找目标实体，这里以单词边界匹配
        if re.search(r'\b{}\b'.format(re.escape(entity)), sentence, re.IGNORECASE):
            return sentence

    return None

def  get_qid(path,candidate_list_origin,text,entity):
    bert_model = BertModel.from_pretrained(f"{path}/model/bert-base-uncased/").to("cuda:0")
    bert_tokenizer = BertTokenizer.from_pretrained(f"{path}/model/bert-base-uncased/")

    # 遍历每个字典对象，取出item里的value值，获得qid，取出description里的value值，得到新的字典对象，装入list
    candidate_list = []
    for item in candidate_list_origin:
        d = {}
        item_value = item['item']['value']
        index = item_value.find('http://www.wikidata.org/entity/')
        if index != -1:
            qid = item_value[index + len('http://www.wikidata.org/entity/'):]
        d['qid'] = qid
        d['description'] = item['description']['value']
        candidate_list.append(d)
    if len(candidate_list) > 1:
        # 找到实体所在原文本里的一小句话
        entity_in_sentence = find_sentence_with_entity(text, entity)
        if entity_in_sentence==None:
            return None
        # 利用bert+词向量
        begin,end = find_word_positions(entity_in_sentence,entity)
        inputs1 = bert_tokenizer(entity_in_sentence,max_length=512,return_tensors="pt", truncation=True, padding=True).to("cuda:0")
        with torch.no_grad():
            outputs1 = bert_model(**inputs1)
        embedding_text_entity = outputs1.last_hidden_state[:, begin:end, :].mean(dim=1)
        max_score = -1
        for item in candidate_list:
            inputs2 = bert_tokenizer(item['description'], return_tensors="pt", truncation=True, padding=True).to("cuda:0")
            with torch.no_grad():
                outputs2 = bert_model(**inputs2)
            embedding_description = outputs2.last_hidden_state.mean(dim=1)  # Average pooling over all tokens
            similarity = cosine_similarity(embedding_text_entity, embedding_description)
            similarity_score = similarity.item()
            if similarity_score>max_score:
                max_score = similarity_score
                qid = item['qid']
        if max_score<0.4:
            return None
        return qid
    elif len(candidate_list) == 1:
        return candidate_list[0]['qid']
    else:
        return None

def get_kg_triplets(args, entity_list,text):
    # 1.为entity_list里面的每个实体进行wikidata实体链接匹配
    # 首先为每个entity通过sparql query在wikidata中找到entity name相同的候选wikidata实体
    result = []
    for entity in entity_list:
        language = 'en'
        if args.dataset=='weibo':
            language='zh'
        s1 = f"""SELECT ?item ?itemLabel ?description
                            WHERE {{
                              ?item rdfs:label "{entity}"@{language}.
                              ?item schema:description ?description.
                              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],{language}". }}
                              FILTER(LANG(?description) = "{language}")
                            }}"""
        candidate_list_origin = sparal_query(s1)
        qid = get_qid(args.path, candidate_list_origin, text, entity)
        if qid==None:
            s1 = f"""SELECT ?item ?itemLabel ?description
                            WHERE {{
                            ?item rdfs:label "The {entity}"@{language}.
                            ?item schema:description ?description.
                            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],{language}". }}
                            FILTER(LANG(?description) = "{language}")
                            }}"""
            # [{'item': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q99509086'}, 'description': {'xml:lang': 'en', 'type': 'literal', 'value': 'artist (born 1971)'}, 'itemLabel': {'xml:lang': 'en', 'type': 'literal', 'value': 'Jack Bauer'}},{}]
            candidate_list_origin = sparal_query(s1)
            qid = get_qid(args.path, candidate_list_origin, text, entity)
            if qid==None:
                continue
        # 2. 找到匹配实体对应的qid了，查出该实体在wikidata里的所有三元关系，存到wikidata_triplets文件夹下，qid.json为名
        # 如果qid.json存在就不再执行操作
        os.makedirs(f"{args.path}/fnd/wikidata_triplets_zh", exist_ok=True)
        os.makedirs(f"{args.path}/fnd/wikidata_triplets", exist_ok=True)
        file_path = f"{args.path}/fnd/wikidata_triplets/{qid}.json"
        if args.dataset=='weibo':
            file_path = f"{args.path}/fnd/wikidata_triplets_zh/{qid}.json"
        if not os.path.exists(file_path):
            step2(args, qid, entity)
        if os.path.exists(file_path):
            # 读取JSON文件,[["IU", "Pinterest username", "iuofficial"], ["IU", "discography", "IU discography"], ["IU", "voice type", "light soprano"]]
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    kg_triplet = json.load(f)
                # result存这个文本所对应的kg_triplets
                result.extend(kg_triplet)
            except json.JSONDecodeError as e:
                # 捕获异常
                print(e)
        else:
            print(f"File '{file_path}' not found.")
            # 处理文件不存在的情况
    unique_data = list(set(map(tuple, result)))
    unique_data = [list(item) for item in unique_data]
    return unique_data


def textualize_graph(triplets):
    nodes = {}
    edges = []
    for tri in triplets:
        src, edge_attr, dst = tri
        if src not in nodes:
            nodes[src] = len(nodes)
        if dst not in nodes:
            nodes[dst] = len(nodes)
        edges.append({'src': nodes[src], 'edge_attr': edge_attr.lower().strip(), 'dst': nodes[dst], })

    nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
    edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])
    return nodes, edges


def step_one(args):
    instruction_ner = "\n抽取文本中可能存在的实体，并以json{'person/location/organization/geopolitics': [entity]}格式输出。"
    instruction_re = "\n抽取文本中可能存在的关系，并以json[{'relation': 'country of citizenship / occupation / eduated at / father / mother / spouse / unmarried partner / child / member of / performed by / award received/present in work / country / located in / nominated for / works', 'head entity': '', 'tail entity': ''},]格式输出."
    if args.dataset == 'weibo':
        instruction_ner = "\n抽取文本中可能存在的实体，并以json{'人名/地名/机构组织名/地缘政治': [实体]}格式输出。"
        instruction_re = "\n抽取文本中可能存在的关系，并以json[{'relation': '国籍/职业/就读于/父亲/母亲/配偶/未婚伴侣/子女/成员/演出者/获得奖项/目前工作于/国家/位于/提名/作品', 'head entity': '', 'tail entity': ''},]格式输出."
    yayi = YaYiNerRe(args.path)
    with open(f'{args.path}/fnd/dataset/{args.dataset}/origin_text.json', 'r',encoding="utf-8") as f:
        data = json.load(f)
    try:
        with open(f"{args.path}/fnd/dataset/{args.dataset}/text_entity_relation.txt", "a", encoding="utf-8") as output:
            for idx, (_, v) in enumerate(tqdm(data.items(), desc="do ner_re")):
                retry=3
                prompt = v['text'][:args.max_length] + instruction_ner
                while retry>0:
                    ner_txt = yayi.ner_re(prompt)
                    # 查找 <reserved_14> 的位置
                    reserved_index = ner_txt.find('<reserved_14>')
                    entity_dict = dict()
                    if reserved_index != -1:
                        # 截取 <reserved_14> 之后的部分
                        ner_json = ner_txt[reserved_index + len('<reserved_14>'):]
                        # 解析 JSON 字符串为 Python 对象
                        try:
                            entity_dict = json.loads(ner_json)
                            if isinstance(entity_dict,dict):
                                break
                            else:
                                retry-=1
                        except json.JSONDecodeError as e:
                            retry-=1
                    else:
                        retry-=1
                retry=3
                prompt = v['text'][:args.max_length] + instruction_re
                while retry>0:
                    re_txt = yayi.ner_re(prompt)
                    # 查找 <reserved_14> 的位置
                    reserved_index = re_txt.find('<reserved_14>')
                    relation_list = []
                    if reserved_index != -1:
                        # 截取 <reserved_14> 之后的部分
                        re_json = re_txt[reserved_index + len('<reserved_14>'):]
                        # 解析 JSON 字符串为 Python 对象
                        try:
                            relation_list = json.loads(re_json)
                            if isinstance(relation_list,list):
                                break
                            else:
                                retry-=1
                        except json.JSONDecodeError as e:
                            retry-=1
                    else:
                        retry-=1

                print(f"{idx}_实体识别信息抽取完成!")
                if not isinstance(entity_dict, dict):
                    entity_dict=dict()
                entity_list=[]
                for key,value in entity_dict.items():
                    entity_list.extend(value)

                if not isinstance(relation_list, list):
                    relation_list=[]
                tmp=dict()
                tmp['text']=v['text']
                tmp['label']=v['label']
                tmp['entity_list']=entity_list
                tmp['relation_list']=relation_list
                output.write(json.dumps(tmp))
                output.write("\n")
                if idx%500==0:
                    output.flush()
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        if output:
            output.close()  # 确保文件关闭

def generate_split(start, end , path):

    indices = np.arange(start,end)
    train_indices, temp_data = train_test_split(indices, test_size=0.4, random_state=42)
    val_indices, test_indices = train_test_split(temp_data, test_size=0.5, random_state=42)
    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    os.makedirs(path, exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/train_indices.txt', 'a') as file:
        file.write('\n')
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/val_indices.txt', 'a') as file:
        file.write('\n')
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/test_indices.txt', 'a') as file:
        file.write('\n')
        file.write('\n'.join(map(str, test_indices)))

def step_two(args):
    prefix_full_path=f"{args.path}/fnd/dataset/{args.dataset}"
    load_qid_relation(args.path)
    load_statement_value(args.path)
    idx = 0
    os.makedirs(prefix_full_path+'/kg', exist_ok=True)
    with open(f"{prefix_full_path}/text_entity_relation.txt", 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            item_json = json.loads(line)
            # 读取entity_list和relation_list
            kg_triplets=[]
            if len(item_json['entity_list'])!=0:
                kg_triplets = get_kg_triplets(args, item_json['entity_list'], item_json['text'][:args.max_length])
                print(f"{idx}_外部知识获取完成!")
                with open(f"{prefix_full_path}/kg/{idx}.txt", 'w') as f:
                    for sublist in kg_triplets:
                        f.write(',;'.join(map(str, sublist)) + '\n')

            news_triplets=[]
            for item in item_json['relation_list']:
                tri = []
                if 'head entity' not in item or 'tail entity' not in item or 'relation' not in item:
                    continue
                tri.append(item['head entity'])
                tri.append(item['relation'])
                tri.append(item['tail entity'])
                news_triplets.append(tri)

            triplets=[]
            if args.graph_name=='all':
                triplets.extend(kg_triplets)
                triplets.extend(news_triplets)
            elif args.graph_name=='onlykg':
                triplets=kg_triplets
            else:
                triplets=news_triplets

            # process triplets
            # 1. 去重
            unique_triplets = list(set(map(tuple, triplets)))

            # 2. 去除对称冗余
            final_triplets = []
            seen_symmetric = set()

            for triplet in unique_triplets:
                e1, rel, e2 = triplet
                symmetric_key = tuple(sorted((e1, e2))) + (rel,)

                if symmetric_key not in seen_symmetric:
                    seen_symmetric.add(symmetric_key)
                    final_triplets.append(triplet)

            os.makedirs(f"{prefix_full_path}/{args.graph_name}", exist_ok=True)
            os.makedirs(f"{prefix_full_path}/{args.graph_name}/nodes", exist_ok=True)
            os.makedirs(f"{prefix_full_path}/{args.graph_name}/edges", exist_ok=True)
            os.makedirs(f"{prefix_full_path}/{args.graph_name}/graphs", exist_ok=True)
            nodes, edges = textualize_graph(final_triplets)
            nodes.to_csv(f"{prefix_full_path}/{args.graph_name}/nodes/{idx}.csv", index=False)
            nodes.to_csv(f"{prefix_full_path}/{args.graph_name}/edges/{idx}.csv", index=False)
            idx+=1

    sample_num = idx

    def _encode_graphs():
        for idx in tqdm(range(sample_num)):
            nodes = pd.read_csv(f"{prefix_full_path}/{args.graph_name}/nodes/{idx}.csv")
            edges = pd.read_csv(f"{prefix_full_path}/{args.graph_name}/edges/{idx}.csv")
            node_attr = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            edge_index = torch.tensor([edges.src, edges.dst]).long()
            pyg_graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
            assert pyg_graph.x.shape[0]==pyg_graph.num_nodes
            torch.save(pyg_graph,f"{prefix_full_path}/{args.graph_name}/graphs/{idx}.pt")

    model, tokenizer, device = load_model['sbert'](args.path)
    text2embedding = load_text2embedding['sbert']

    _encode_graphs()

if __name__ == '__main__':
    args = parse_args()
    step_one(args)
    step_two(args)
    if args.dataset == 'politifact':
        generate_split(0,317, f"{args.path}/fnd/dataset/{args.dataset}/split")
        generate_split(317, 703, f"{args.path}/fnd/dataset/{args.dataset}/split")
    if args.dataset == 'gossipcop':
        generate_split(0,3421, f"{args.path}/fnd/dataset/{args.dataset}/split")
        generate_split(3421, 14579, f"{args.path}/fnd/dataset/{args.dataset}/split")
    if args.dataset == 'weibo':
        generate_split(0,4641, f"{args.path}/fnd/dataset/{args.dataset}/split")
        generate_split(4641, 9592, f"{args.path}/fnd/dataset/{args.dataset}/split")