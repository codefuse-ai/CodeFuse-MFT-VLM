import argparse
import torch
import os
import os.path as osp
import json
from tqdm import tqdm
import threading
import pandas as pd
import base64
import io
import random
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model, load_pretrained_model_custom_proj, load_mixed_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image

#def eval_model(args, questions, start, end, ans_file, lock):
def eval_model(args, questions, start, end, ans_file):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    #model_name = get_model_name_from_path(model_path) + "-lora"
    model_name = "yuque_qwen-7b-lora"
    #tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    #tokenizer, model, image_processor, context_len = load_pretrained_model_custom_proj(model_path, args.model_base, model_name, args.mm_projector)
    tokenizer, model, image_processor, context_len = load_mixed_pretrained_model(model_path, args.model_base, model_name, args.vision_tower, args.mm_projector_type, args.mm_projector)
    tokenizer.pad_token_id = tokenizer.eod_id
    model = model.cuda()

    output_dict = {"index": [], 'question': [], "A": [], "B": [], "C": [], "D": [], "prediction": []}

    index = 0
    total_cnt = 0
    correct_cnt = 0

    for i in tqdm(range(start, end), desc="{} ({} - {})".format(threading.current_thread().name, start, end)):
        if i % 100 == 0 and i != 0:
            print("correct cnt", correct_cnt)
            print("Total cnt", total_cnt)
            print("accuracy", correct_cnt / total_cnt)
        line = questions.iloc[i]
        idx = line["index"]
        image = decode_base64_to_image(line["image"])
        question = str(line['question'])
        hint = str(line['hint'])
        
        #### English or Chinese
        #question = "<|im_start|>user\n" + "Picture 1:<img></img>\n" + hint + " " + question + " Please choose a correct option in the following choice.\n"  
        question = "<|im_start|>user\n" + "Picture 1:<img></img>\n" + hint + " " + question + " 请从以下选项中选择一个正确选项。\n"  
        
        output_dict['index'].append(idx)
        output_dict['question'].append(line['question'])
        output_dict['A'].append(line['A'])
        output_dict['B'].append(line['B'])
        output_dict['C'].append(line['C'])
        output_dict['D'].append(line['D'])


        im_start = torch.tensor(tokenizer.im_start_id)  ##每次对话起始符，无论用户还是机器
        im_end = torch.tensor(tokenizer.im_end_id)  ##每次对话起始符，无论用户还是机器
        nl_tokens = torch.tensor(tokenizer('\n').input_ids)
        _system = torch.tensor(tokenizer('system').input_ids)  ##全样本就一个的system
        _user = torch.tensor(tokenizer('user').input_ids)
        _assistant = torch.tensor(tokenizer('assistant').input_ids)

        inputs = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + question
        if str(line['A']) != 'nan':
            inputs = inputs + "A. " + str(line['A']) + "\n"
        if str(line['B']) != 'nan':
            inputs = inputs + "B. " + str(line['B']) + "\n"
        if str(line['C']) != 'nan':
            inputs = inputs + "C. " + str(line['C']) + "\n"
        if str(line['D']) != 'nan':
            inputs = inputs + "D. " + str(line['D']) + "\n"
        inputs += "<|im_end|>\n"
        inputs += "<|im_start|>assistant\n"


        tokens = tokenizer(
            inputs,
            max_length=tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.cuda()
        
        #image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        #stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stop_str = tokenizer.pad_token
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).cuda(),
                do_sample=True,
                temperature=0.2,
                top_p=0.3,
                top_k=0,
                #num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=2048,
                return_dict_in_generate=False,
                use_cache=True)
        
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        output_text = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        #import pdb; pdb.set_trace()
        output_text = output_text.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("\n", "")
        
        output_dict['prediction'].append(output_text)

        if output_text[0] == line['answer']:
            correct_cnt += 1
        elif line['answer'] in ["A", "B", "C", "D"] and line[line['answer']] in output_text:
            correct_cnt += 1
        total_cnt += 1

        #import pdb; pdb.set_trace()

        index += 1
    
    print("Final correct cnt", correct_cnt)
    print("Final Total cnt", total_cnt)
    print("Final accuracy", correct_cnt / total_cnt)
    out_df = pd.DataFrame(output_dict)
    out_df.to_csv(ans_file, sep="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--mm-projector-type", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    args = parser.parse_args()

    thread_num = 1

    # questions file
    questions = pd.read_csv(args.question_file, delimiter='\t')

    # answers file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = answers_file

    #eval_model(args, questions, 0, len(questions), ans_file)
    eval_model(args, questions, 0, 20, ans_file)