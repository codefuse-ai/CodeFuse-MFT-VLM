import argparse
import torch
import os
import os.path as osp
import json
from tqdm import tqdm
import threading
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

    index = 0
    for i in tqdm(range(start, end), desc="{} ({} - {})".format(threading.current_thread().name, start, end)):
        line = questions[i]
        idx = line["id"]
        image_file = line["image"]
        qs = line["conversations"][0]["value"].replace("{}\n".format(DEFAULT_IMAGE_TOKEN), "")
        cur_prompt = qs
        #if model.config.mm_use_im_start_end:
        #qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        #else:
        #    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        #conv = conv_templates[args.conv_mode].copy()
        #conv.append_message(conv.roles[0], qs)
        #conv.append_message(conv.roles[1], None)
        #prompt = conv.get_prompt()

        ori_ir = line["conversations"][1]["value"]

        for sentence in line['conversations']:
            if "Qwen" in args.model_base:
                if sentence['from']=='human':
                    if DEFAULT_IMAGE_TOKEN in sentence['value']:
                        content = "<|im_start|>user\n" + "Picture 1:<img></img>\n" + sentence['value'].strip(DEFAULT_IMAGE_TOKEN) + "<|im_end|>\n"
                    else:
                        content = "<|im_start|>user\n" + sentence['value'].strip(DEFAULT_IMAGE_TOKEN) + "<|im_end|>\n"
                else: 
                    content = "<|im_start|>assistant\n" + sentence['value'].strip(DEFAULT_IMAGE_TOKEN) + "<|im_end|>\n"
                sentence['value'] = content

        im_start = torch.tensor(tokenizer.im_start_id)  ##每次对话起始符，无论用户还是机器
        im_end = torch.tensor(tokenizer.im_end_id)  ##每次对话起始符，无论用户还是机器
        nl_tokens = torch.tensor(tokenizer('\n').input_ids)
        _system = torch.tensor(tokenizer('system').input_ids)  ##全样本就一个的system
        _user = torch.tensor(tokenizer('user').input_ids)
        _assistant = torch.tensor(tokenizer('assistant').input_ids)

        inputs = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + line["conversations"][0]["value"] + "<|im_start|>assistant\n"

        tokens = tokenizer(
            inputs,
            max_length=tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.cuda()
        
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
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
        first_curly_bracket_idx = None
        for i in range(len(output_text)):
            if output_text[i] == "{" and output_text[i + 1: i + 7] == "data":
                first_curly_bracket_idx = i
                break
        if first_curly_bracket_idx is not None:
            output_text = output_text[i:]
        #import pdb; pdb.set_trace()

        #input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        #stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        #keywords = [stop_str]
        #stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        #with torch.inference_mode():
        #    output_ids = model.generate(
        #        input_ids,
        #        images=image_tensor.unsqueeze(0).half().cuda(),
        #        do_sample=True,
        #        temperature=args.temperature,
        #        top_p=args.top_p,
        #        num_beams=args.num_beams,
        #        # no_repeat_ngram_size=3,
        #        max_new_tokens=2048,
        #        use_cache=True)
        


        #input_token_len = input_ids.shape[1]
        #n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        #if n_diff_input_output > 0:
        #    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        #outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        #outputs = outputs.strip()
        #if outputs.endswith(stop_str):
        #    outputs = outputs[:-len(stop_str)]
        #outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        #with lock:
        ans_file.write(json.dumps({"question_id": idx,
                                       "from": line["from"],
                                       "prompt": cur_prompt,
                                       "ir": ori_ir,
                                       "text": output_text,
                                       "answer_id": ans_id,
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")
        ans_file.flush()
        index += 1

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
    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.loads(f.read())
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    n = len(questions)
    step = n // thread_num

    # answers file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    eval_model(args, questions, 0, 100, ans_file)

    #threads = []
    #lock = threading.Lock()
    #for i in range(thread_num):
    #    start = i * step
    #    end = n if i == 3 else (i + 1) * step
    #    t = threading.Thread(target=eval_model, args=(args, questions, start, end, ans_file, lock))
    #    threads.append(t)

    #for t in threads:
    #    t.start()

    #for t in threads:
    #    t.join()

    ans_file.close()