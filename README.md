## DSTA

We propose a KG-enhanced LLM framework for fake news Detection with Semantic and Topological Awareness, abbreviated as **DSTA**, by combining KGs, GNNs and LLMs. It mainly consists of four modules, i.e., news structure extraction, fact enhancement via KGs, text-graph modality alignment, and detection through LLMs.
<img src="https://github.com/shi-caicai/DSTA/blob/master/figs/framework.png" style="zoom: 70%;" />

### Environment

```
conda create --name DSTA python=3.9 -y
conda activate DSTA

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

pip install peft
pip install pandas
pip install ogb
pip install transformers
pip install wandb
pip install sentencepiece
pip install torch_geometric
pip install datasets
pip install pcst_fast
pip install gensim
pip install scipy==1.12
pip install protobuf
```



### Preparation

1. Download the models [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased), [sentence-transformers](https://huggingface.co/sentence-transformers/all-roberta-large-v1), and [YAYI](https://huggingface.co/wenge-research/yayi-uie), then place them in src/model/.
2. Download the experiment [dataset](https://www.kaggle.com/datasets/shicaicaiya/dsta-experiment-dataset) including the directory and place it in fnd/.

3. Download the LLM like glm4-9b and place it in the src/model/, modify the LLM path in the src/model/init.py file.



### Experiment

step1:

```
python step1.py --path your_project_path --dataset gossipcop --graph_name all
```

step2:

```
python train.py --path your_project_path --dataset gossipcop --llm_model_name glm4-9b --llm_frozen False --gnn_model_name gat --rationale_yes_no False
```

