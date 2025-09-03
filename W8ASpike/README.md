---
frameworks:
- Pytorch
license: Apache License 2.0
tasks:
- text-generation

#model-type:
##such as  gpt、phi、llama、chatglm、baichuan, etc.
#- gpt

#domain:
##such as  nlp、cv、audio、multi-modal, etc.
#- nlp

#language:
##language code list https://help.aliyun.com/document_detail/215387.html?spm=a2c4g.11186623.0.0.9f8d7467kni6Aa
#- cn 

#metrics:
##such as  CIDEr、Blue、ROUGE, etc.
#- CIDEr

#tags:
##various custom tags, including pretrained, fine-tuned, instruction-tuned, RL-tuned, and others
#- pretrained

#tools:
##such as  vllm、fastchat、llamacpp、AdaSeq, etc.
#- vllm
---
### You are viewing the default Readme template as no detailed model-card was provided by the model’s contributors. You can access the model files in the "Files and versions" tab.
#### Model files may be downloaded with ModelScope SDK or through git clone directly.

Download with ModelScope’s Python SDK
```bash
#Install ModelScope
pip install modelscope
```
```python
#Download with ModelScope’s Python SDK
from modelscope import snapshot_download
model_dir = snapshot_download('Abel2076/SpikingBrain-76B-W8ASpike')
```
Download with Git clone
```
git clone https://www.modelscope.cn/Abel2076/SpikingBrain-76B-W8ASpike.git
```

<p style="color: lightgrey;">If you are a contributor to this model, we invite you to promptly update the model card content according to <a href="https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88" style="color: lightgrey; text-decoration: underline;">the model contribution documentation</a>.</p>