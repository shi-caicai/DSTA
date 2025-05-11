import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

class YaYiNerRe:
    def __init__(self,path):
        self.tokenizer = AutoTokenizer.from_pretrained(path+"/model/yayi/",
                                                  trust_remote_code=True,from_tf=True)
        self.model = AutoModelForCausalLM.from_pretrained(path+"/model/yayi/",
                                                     torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0")
    def ner_re(self,prompt):
        prompt = "<reserved_13>" + prompt + "<reserved_14>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        response = self.model.generate(**inputs, max_new_tokens=512, temperature=0)
        return self.tokenizer.decode(response[0], skip_special_tokens=True)