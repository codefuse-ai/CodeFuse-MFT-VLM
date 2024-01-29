# 这是容器的初始化文件，此脚本会在容器启动后运行，可以在此写上常用包的安装脚本，例如：pip install torch
#pip install deepspeed==0.8.3
#pip install transformers==4.30.0
#pip install accelerate==0.20.3
#pip install /mnt/user/qumu/libs/peft-662ebe593e5d4a2d64a4ee0a0c61c807f7a62617
#pip install BitsAndBytes==0.39.0
#pip install xformers
#pip install ujson
#pip install jsonlines

pip install SentencePiece==0.1.99  -i https://pypi.antfin-inc.com/simple/
pip install alps==2.3.1.8  -i https://pypi.antfin-inc.com/simple/
pip install deepspeed==0.9.5 -i https://pypi.antfin-inc.com/simple/
pip install accelerate==0.23.0 -i https://pypi.antfin-inc.com/simple/
pip install transformers==4.32.0 -i https://pypi.antfin-inc.com/simple/
pip install peft==0.5.0 -i https://pypi.antfin-inc.com/simple/
pip install tiktoken==0.5.1 -i https://pypi.antfin-inc.com/simple/
pip install transformers_stream_generator==0.0.4 -i https://pypi.antfin-inc.com/simple/