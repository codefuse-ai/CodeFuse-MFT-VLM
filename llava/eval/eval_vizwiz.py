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

    ans_fp = open(ans_file, "w")

    total_cnt = 0
    correct_cnt = 0

    i = 0
    for ln in questions:
        if i % 100 == 0 and i != 0:
            print(i, "correct cnt", correct_cnt)
            print(i, "Total cnt", total_cnt)
            print(i, "accuracy", correct_cnt / total_cnt)
        
        #ln = next(questions)
        question = json.loads(ln)

        image_fn = osp.join(args.image_folder, "/".join(question['image'].split("/")[3:]))
        
        if not osp.isfile(image_fn):
            continue
        image = Image.open(image_fn).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        im_start = torch.tensor(tokenizer.im_start_id)  ##每次对话起始符，无论用户还是机器
        im_end = torch.tensor(tokenizer.im_end_id)  ##每次对话起始符，无论用户还是机器
        nl_tokens = torch.tensor(tokenizer('\n').input_ids)
        _system = torch.tensor(tokenizer('system').input_ids)  ##全样本就一个的system
        _user = torch.tensor(tokenizer('user').input_ids)
        _assistant = torch.tensor(tokenizer('assistant').input_ids)

        inputs = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        prompt = "<|im_start|>user\n" + "Picture 1:<img></img>\n" + question['question'] + "<|im_end|>\n" + "<|im_start|>assistant\n"
        inputs += prompt

        tokens = tokenizer(
            inputs,
            max_length=tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.cuda()

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

        output_text = output_text.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("\n", "")
        output_text = output_text.lower()
        #if i % 20 == 0:
        #    import pdb; pdb.set_trace()

        answer = question['answer']
        if answer in output_text:
            correct_cnt += 1
        total_cnt += 1

        out_j_dict = {}
        for k in question:
            out_j_dict[k] = question[k]
        out_j_dict['prediction'] = output_text

        out_j_str = json.dumps(out_j_dict) + '\n'
        ans_fp.write(out_j_str)
        i += 1

    print("Final correct cnt", correct_cnt)
    print("Final Total cnt", total_cnt)
    print("Final accuracy", correct_cnt / total_cnt)
    ans_fp.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
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
    questions = open(args.question_file)

    # answers file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = answers_file

    #eval_model(args, questions, 0, len(questions), ans_file)
    eval_model(args, questions, 0, 20, ans_file)
        