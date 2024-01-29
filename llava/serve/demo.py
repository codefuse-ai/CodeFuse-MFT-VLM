#!/usr/bin/env python
from __future__ import print_function

import sys
import argparse

from os.path import basename
from classes.Utils import *
from classes.Compiler import *
import os.path as osp

FILL_WITH_RANDOM_TEXT = True
TEXT_PLACE_HOLDER = "[]"


def render_content_with_text(key, value):
    if FILL_WITH_RANDOM_TEXT:
        if key.find("btn") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text())
        elif key.find("title") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=5, space_number=0))
        elif key.find("text") != -1:
            value = value.replace(TEXT_PLACE_HOLDER,
                                  Utils.get_random_text(length_text=56, space_number=7, with_upper_case=False))
    return value

def render_dsls(input_fp, args):
    i = 0
    for ln in input_fp:
        compiler = Compiler(args.dsl_path)
        j_dict = json.loads(ln)
        input_text = j_dict['prediction']
        input_text = input_text.replace("{", "{\n").replace("}", "\n}\n").replace("\n\n", '\n').rstrip("\n")

        label = j_dict['label'][0]
        label = label.rstrip("\n")

        output_file_path = osp.join(args.output_folder, j_dict['fn'].split("/")[-1].replace(".gui", ".html"))
        pred_output_html = compiler.compile(input_text, output_file_path, rendering_function=render_content_with_text)
        #label_output_html = compiler.compile(label, output_file_path, rendering_function=render_content_with_text)
        i += 1
        #print(pred_output_html)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="input.jsonl")
    parser.add_argument("--output-folder", type=str, default="output/")
    parser.add_argument("--dsl-path", type=str, default="dsl.json")
    args = parser.parse_args()
    
    input_fp = open(args.input_file)
    render_dsls(input_fp, args)