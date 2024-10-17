## CodeFuse-VLM
CodeFuse-VLM is a Multimodal LLM(MLLM) framework that provides users with multiple vision encoders, multimodal alignment adapters, and LLMs. Through CodeFuse-VLM framework, users are able to customize their own MLLM model to adapt their own tasks.
As more and more models are published on Huggingface community, there will be more open-source vision encoders and LLMs. Each of these models has their own specialties, e.g. Code-LLama is good at code-related tasks but has poor performance for Chinese tasks. Therefore, we built CodeFuse-VLM framework to support multiple vision encoders, multimodal alignment adapters, and LLMs to adapt different types of tasks.
![img.jpg](./CodeFuse-VLM-arch.png)

Under CodeFuse-VLM framework, we use cross attention multimodal adapter, Qwen-14B LLM, and Qwen-VL's vision encoder to train CodeFuse-VLM-14B model. On multiple benchmarks, our CodeFuse-VLM-14B shows superior performances over Qwen-VL and LLAVA-1.5.
![img.jpg](./CodeFuse-VLM-14B-performance.png)

Here is the table for different MLLM model's performance on benchmarks
Model | MMBench | MMBench-CN | VqaV2 | GQA | TextVQA | Vizwiz
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
LLAVA-1.5 | 67.7 | 63.6 | 80.0 | 63.3 | 61.3 | 53.6
Qwen-VL | 60.6 | 56.7 | 78.2 | 57.5 | 63.8 | 38.9 
CodeFuse-VLM-14B | 75.7 | 69.8 | 79.3 | 59.4 | 63.9 | 45.3

Our model achieved high ranking on MMBenchmark: https://mmbench.opencompass.org.cn/leaderboard

Here's our model's demo video

https://private-user-images.githubusercontent.com/22836551/300386230-8e64f615-ac0e-447e-9695-c96b254d484f.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDY1MjExODksIm5iZiI6MTcwNjUyMDg4OSwicGF0aCI6Ii8yMjgzNjU1MS8zMDAzODYyMzAtOGU2NGY2MTUtYWMwZS00NDdlLTk2OTUtYzk2YjI1NGQ0ODRmLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTI5VDA5MzQ0OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQ5NzNjM2U1ZWU4NDU0Yzc5NmE4ZTM1NzY2ZjU4YjRjY2ZhNjMzODk0ZDgzMDg4N2FjYjZhYTllM2E3NTAyMWQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.pr-ad7rKYBgk26DTItj2q2q9I5dRWnBNHbV9M7GSVCo


## Contents
- [Install](#Install)
- [Datasets](#Datasets)
- [Multimodal Alignment](#Multimodal-Alignment)
- [Visual Instruction Tuning](#Visual-Instruction-Tuning)
- [Evaluation](#Evaluation)

## Install
Please run sh init\_env.sh 

## Datasets
Here's the table of datasets we used to train CodeFuse-VLM-14B:

Dataset | Task Type | Number of Samples 
| ------------- | ------------- | ------------- |
synthdog-en | OCR | 800,000
synthdog-zh	| OCR | 800,000
cc3m(downsampled)| Image Caption | 600,000
cc3m(downsampled)| Image Caption | 600,000
SBU | Image Caption | 850,000
Visual Genome VQA (Downsampled) | Visual Question Answer(VQA) | 500,000
Visual Genome Region descriptions (Downsampled) | Reference Grouding | 500,000
Visual Genome objects (Downsampled) | Grounded Caption | 500,000
OCR VQA (Downsampled) | OCR and VQA | 500,000

Please download these datasets on their own official websites.

## Multimodal Alignment
Please run sh scripts/pretrain.sh or sh scripts/pretrain\_multinode.sh

## Visual Instruction Tuning
Please run sh scripts/finetune.sh or sh scripts/finetune\_multinode.sh

## Evaluation
Please run python scripts in directory llava/eval/. Our pre-trained CodeFuse-VLM-14B can be loaded with the following code:

```
import os
from llava.model.builder import load_mixed_pretrained_model

model_path = '/pretrained/model/path'
tokenizer, model, image_processor, context_len = load_mixed_pretrained_model(model_path, None, 'qwen-vl-14b', os.path.join(model_path, 'Qwen-VL-visual'), 'cross_attn', os.path.join(model_path, 'mm_projector/mm_projector.bin'))
```

You can also run scripts/merge\_qwen\_vl\_weights.sh first and load the merged model by the following code:

```
from llava.model import LlavaQWenForCausalLM

model = LlavaQWenForCausalLM.from_pretrained('/path/to/our/pretrained/model')
```

## CodeFuse-VLM Product Video
Here's the demo video of front-end code copilot backed by our VLM model

https://private-user-images.githubusercontent.com/22836551/300398424-201f667d-6b6b-4548-b3e6-724afc4b3071.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDY1MjE5MTIsIm5iZiI6MTcwNjUyMTYxMiwicGF0aCI6Ii8yMjgzNjU1MS8zMDAzOTg0MjQtMjAxZjY2N2QtNmI2Yi00NTQ4LWIzZTYtNzI0YWZjNGIzMDcxLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTI5VDA5NDY1MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWI0ZmJmZWNlNDZmNWM3NzA0OThlMmY1ODY4MDkxNWY5ZWNiNzRiYjJkYmE4NjEzM2EwYWRiNWY2ODc3N2ViYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.BIvWGNx0XV7RoauxB0c2noEdbfZfu8-16LPHtCaCJ9k

## Join US
<img src="./CodeFuse_UserGroup.png" alt="img" width="200" height="200">

We are the Risk Intelligence team within the Platform Technology Business Group at Ant Group, dedicated to the intelligentization of Ant Group's platform engineering. Established for over three years, our team has played a pivotal role in supporting the intelligent operation and maintenance of Ant Group's cloud computing infrastructure. Our mission is to build algorithm services and platforms with a wide user base through world-class technological innovation and impact, supporting the implementation of internal and external products and businesses.
Embracing an innovation-driven ethos, our team not only supports business implementation but also propels technological influence. Over the past three years, we have published more than 20 papers at top conferences like ICLR, NeurIPS, KDD, and ACL. Our innovative business outcomes have earned us two Ant Technology's highest T-Star awards and one SuperMA award from Ant Group. Our open-source project CodeFuse has received 4K stars as of February 2024, and our models have been downloaded over 1.5 million times on Huggingface and Modelscope.

**We are on the lookout for top talents to join our vibrant team! If you're eager to develop your career in an environment filled with energy, innovation, and a culture of excellence, we welcome you to explore our career opportunities for both campus and experienced hires. Join us and be a part of creating the next milestone in the industry.**

**Campus Recruitment**: https://hrrecommend.antgroup.com/guide.html?code=8uoP5mlus5DqQYbE_EnqcE2FD5JZH21MwvMUIb9mb6X3osXPuBraG54SyM8GLn_7

**Experienced Hires**: https://talent.antgroup.com/off-campus-position?positionId=1933830

我们是平台技术事业群风险智能团队，负责蚂蚁蚂蚁集团平台工程的智能化，团队成立3年多以来，支持了蚂蚁集团云计算基础设施智能化运维的升级改造。团队的Mission是，通过世界级的技术创新和影响，构建有广泛用户的算法服务和平台，支撑内外部产品和业务落地。团队秉承创新基因，在支撑业务落地的同时，推动技术影响。3年以来在ICLR、NeurIPS、KDD、ACL等顶会发表论文20余篇，创新业务结果获得两次蚂蚁技术最高奖T-Star，1次蚂蚁集团最高奖SuperMA。开源项目CodeFuse获得4K点赞(2024年2月)，Huggingface和modelscope上模型累积下载量超过150万次。

**我们正在寻找行业中的佼佼者加入我们的团队！如果您希望在一个充满活力、创新和卓越文化的环境中发展您的职业生涯，欢迎您查看我们的社招&校招机会，加入我们，一起创造下一个行业里程碑。**

**校招**：https://hrrecommend.antgroup.com/guide.html?code=8uoP5mlus5DqQYbE_EnqcE2FD5JZH21MwvMUIb9mb6X3osXPuBraG54SyM8GLn_7

**社招**：https://talent.antgroup.com/off-campus-position?positionId=1933830

