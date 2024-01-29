import argparse
import json
import os

import openai
import time

from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex
import requests
import ast

def aes_encrypt(data, key):
    """aes加密函数，如果data不是16的倍数【加密文本data必须为16的倍数！】，那就补足为16的倍数
    :param key:
    :param data:
    """
    iv = "1234567890123456"
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))  # 设置AES加密模式 此处设置为CBC模式
    block_size = AES.block_size

    # 判断data是不是16的倍数，如果不是用b'\0'补足
    if len(data) % block_size != 0:
        add = block_size - (len(data) % block_size)
    else:
        add = 0
    data = data.encode('utf-8') + b'\0' * add
    encrypted = cipher.encrypt(data)  # aes加密
    result = b2a_hex(encrypted)  # b2a_hex encode  将二进制转换成16进制
    return result.decode('utf-8')

def aes_decode(data, key):
    """aes解密
    :param key:
    :param data:
    """
    iv = '1234567890123456'
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
    result2 = a2b_hex(data)  # 十六进制还原成二进制
    decrypted = cipher.decrypt(result2)
    return decrypted.rstrip(b'\0')  # 解密完成后将加密时添加的多余字符'\0'删除

NUM_SECONDS_TO_SLEEP = 0.5


def get_eval(content: str, max_tokens: int):
    serviceName = "your service name"
    visitDomain = "your domain"
    visitBiz = "your visit biz"
    visitBizLine = "your bizline"
    api_key = "your api key"
    key = "your key"
    param = {
        "serviceName": serviceName,
        "visitDomain": visitDomain,
        "visitBiz": visitBiz,
        "visitBizLine": visitBizLine,
        "cacheInterval": -1,
        "queryConditions": {
            "model": "gpt-3.5-turbo-16k",
            "api_key": api_key,

            
            "messages": [{"role": "user", "content": content}]
        }
    }
    url = 'your url'
    data = json.dumps(param) % url.encode('utf8')
    key = key  # 密钥
    str = aes_encrypt(data, key)
    post_data = {
        "encryptedParam": str
    }
    headers = {
        'Content-Type': 'application/json'
    }
    while True:
        try:
            response = requests.post(url, data=json.dumps(post_data), headers=headers)
            x = response.json()["data"]["values"]["data"]
            ast_str = ast.literal_eval("'" + x + "'")

            js = ast_str.replace('&quot;', '"')
            js = js.replace("&#39;", "'")
            data = json.loads(js)

            ret = data["choices"][0]["message"]["content"]
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)
    return ret


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(', ', " ").replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')

    context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
    image_to_context = {context['image']: context for context in context_list}

    handles = []
    idx = 0
    for ques_js, ans1_js, ans2_js in zip(f_q, f_ans1, f_ans2):
        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)

        inst = image_to_context[ques['image']]

        if isinstance(inst['caption'], list):
            cap_str = '\n'.join(inst['caption'])
        else:
            cap_str = inst['caption']

        
        content = (f'[Context]\n{cap_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[assistant 1]\n{ans1["answer"]}\n\n[End of assistant 1]\n\n'
                   f'[assistant 2]\n{ans2["text"]}\n\n[End of assistant 2]\n\n'
                   f'[System]\nYour are a judge to judge the 2 answers given to you. Please rate each answer between score 0 and 100, and use comma to separate the 2 scores.\n\n')
        cur_js = {
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', ans2['answer_id']),
            'category': "LLAVA"
        }
        if idx >= len(cur_reviews):
            review = get_eval(content, args.max_tokens)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        idx += 1
        print(idx)
    review_file.close()
