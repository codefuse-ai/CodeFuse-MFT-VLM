<div align="center">
  <center><h1>CodeFuse-VLM</h1></center>
</div>

[**English** ](https://github.com/codefuse-ai/CodeFuse-MFT-VLM/blob/master/README.md)| [**HuggingFace** ](https://huggingface.co/codefuse-ai/CodeFuse-VLM-14B)|[ **ModelScope** ](https://modelscope.cn/models/ss41979310/CodeFuse-VLM-14B/summary)
</div>

CodeFuse-VLM 是一个多模态大语言模型框架，该框架为用户提供多种视觉编码器，模态对齐模块和大语言模型的选择，以适配用户对不同任务的需求。

随着huggingface开源社区的不断更新，会有更多的vision encoder 和 LLM 底座发布，这些vision encoder 和 LLM底座都有各自的强项，例如 code-llama 适合生成代码类任务，但是不适合生成中文类的任务；因此我们搭建了CodeFuse-VLM 框架，支持多种视觉模型和语言大模型，使得CodeFuse-VLM可以适应不同种类的任务。

![img.jpg](./CodeFuse-VLM-arch.png)

我们在CodeFuse-VLM 框架下, 使用Qwen-VL的视觉编码器, cross attention模态对齐模块, 和 Qwen-14B 模型训练了 CodeFuse-VLM-14B

CodeFuse-VLM-14B 在多个benchmarks 上的性能超过了Qwen-VL和LLAVA-1.5
![img.jpg](./CodeFuse-VLM-14B-performance.png)

各个模型得分如下表所示:
模型 | MMBench | MMBench-CN | VqaV2 | GQA | TextVQA | Vizwiz
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
LLAVA-1.5 | 67.7 | 63.6 | 80.0 | 63.3 | 61.3 | 53.6
Qwen-VL | 60.6 | 56.7 | 78.2 | 57.5 | 63.8 | 38.9 
CodeFuse-VLM-14B | 75.7 | 69.8 | 79.3 | 59.4 | 63.9 | 45.3

这是我们模型的展示视频

https://private-user-images.githubusercontent.com/22836551/300386230-8e64f615-ac0e-447e-9695-c96b254d484f.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDY1MjExODksIm5iZiI6MTcwNjUyMDg4OSwicGF0aCI6Ii8yMjgzNjU1MS8zMDAzODYyMzAtOGU2NGY2MTUtYWMwZS00NDdlLTk2OTUtYzk2YjI1NGQ0ODRmLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTI5VDA5MzQ0OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQ5NzNjM2U1ZWU4NDU0Yzc5NmE4ZTM1NzY2ZjU4YjRjY2ZhNjMzODk0ZDgzMDg4N2FjYjZhYTllM2E3NTAyMWQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.pr-ad7rKYBgk26DTItj2q2q9I5dRWnBNHbV9M7GSVCo


## Contents
- [Install](#Install)
- [Datasets](#Datasets)
- [Multimodal Alignment](#Multimodal-Alignment)
- [Visual Instruction Tuning](#Visual-Instruction-Tuning)
- [Evaluation](#Evaluation)

## Install
请执行 sh init\_env.sh 

## Datasets
使用了以下数据集训练模型:

数据集 | 任务种类 | 样本量 
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

请到各个数据集的官网上下载这些数据。

## Multimodal Alignment
请执行 sh scripts/pretrain.sh 或者 sh scripts/pretrain\_multinode.sh


## Visual Instruction Tuning
请执行 sh scripts/finetune.sh 或者 sh scripts/finetune\_multinode.sh

## Evaluation
请执行 llava/eval/ 当中的python脚本

## CodeFuse-VLM 产品视频
这是我们模型支持的产品的视频

https://private-user-images.githubusercontent.com/22836551/300398424-201f667d-6b6b-4548-b3e6-724afc4b3071.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDY1MjE5MTIsIm5iZiI6MTcwNjUyMTYxMiwicGF0aCI6Ii8yMjgzNjU1MS8zMDAzOTg0MjQtMjAxZjY2N2QtNmI2Yi00NTQ4LWIzZTYtNzI0YWZjNGIzMDcxLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTI5VDA5NDY1MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWI0ZmJmZWNlNDZmNWM3NzA0OThlMmY1ODY4MDkxNWY5ZWNiNzRiYjJkYmE4NjEzM2EwYWRiNWY2ODc3N2ViYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.BIvWGNx0XV7RoauxB0c2noEdbfZfu8-16LPHtCaCJ9k