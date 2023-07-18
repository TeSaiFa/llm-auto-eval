import os


import time
import argparse
from tqdm import tqdm

from src.prompt.reasoning_QA import get_reasoning_prompt
from src.prompt.multi_choice_QA import get_multi_choice_prompt
from src.prompt.judgement_QA import get_judgement_prompt
from src.prompt.open_QA import get_open_qa_prompt

from src.gptq_inference import get_gptq_model_and_tokenizer
from src.moss_inference import get_moss_model_and_tokenizer, moss_inference
from src.baichuan_inference import get_baichuan_model_and_tokenizer
from src.chatglm_inference import get_glm_model_and_tokenizer, glm_inference
from src.llama_inference import get_llama_model_and_tokenizer

import torch
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList


from loguru import logger

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

def get_prompt_by_typ(data):
    class_1, class_2 = data['一级分类'], data['二级分类']
    typ = data['题型分类']
    domain = data['domain']
    question = data['prompt']
    question = question.strip('\n')
    if domain == '-':
        domain = ''

    if typ in ['推理', '证明', '应用']:
        prompt = get_reasoning_prompt(class_1, class_2, domain, question, typ,)
    if typ == '选择':
        prompt = get_multi_choice_prompt(class_1, class_2, domain, question)
    if typ == '判断':
        prompt = get_judgement_prompt(class_2, question)
    if typ == "问答":
        prompt = get_open_qa_prompt(class_1, class_2, domain, question, typ,)

    prompt = prompt.strip('\n')+'\n'

    return prompt


def inference(args, model, tokenizer, prompt, logits_processor=None):
    if args.model_type == 'moss':
        res = moss_inference(args, model, tokenizer, prompt, logits_processor=logits_processor)
        return res
    if args.model_type == 'glm':
        res = glm_inference(args, model, tokenizer, prompt, logits_processor=logits_processor)
        return res
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:{}".format(args.device))
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=args.do_sample,
            max_length=args.max_length,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            return_dict_in_generate=True,
            repetition_penalty=args.rep_penalty,
            logits_processor=logits_processor if logits_processor else LogitsProcessorList()
        )
    res = tokenizer.decode(generated_ids.sequences[0][len(input_ids[0]):])
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama', help='model to load')
    parser.add_argument('--model_path', type=str, default='/home/fm001/wangyuxuan/models/LLaMA-7B-hf', help='model to load')
    parser.add_argument('--model_type', type=str, default='llama', help='model type to load: llama|glm|moss|baichuan|gptq ')
    parser.add_argument('--data_path', type=str, default='../dataset/test_samples.json', help='dataset path')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='The maximum length of the sequence to be generated.')

    parser.add_argument('--top_p',
                        type=float,
                        default=0.95,
                        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.')
    parser.add_argument('--top_k',
                        type=int,
                        default=40, )
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='The value used to module the next token probabilities.')
    parser.add_argument('--device', type=int, default=0,
                        help='The device used to load the model when using safetensors. Default device is "cpu" or specify, 0,1,2,3,... for GPU device.')
    parser.add_argument('--rep_penalty', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--do_batch', action='store_true')
    parser.add_argument('--use_prefix', action='store_true')
    parser.add_argument('--use_fastllm', action='store_true')


    args = parser.parse_args()

    import json

    data = json.load(open(args.data_path, 'r'))
    results = []
    logger.add(f'log/runtime_{time}_{args.model_name}_{args.data_path.split("/")[-1].strip(".json")}.log')

    logits_processor = [InvalidScoreLogitsProcessor()]
    if args.model_type == 'llama':
        model, tokenizer = get_llama_model_and_tokenizer(args)
    elif args.model_type == 'moss':
        model, tokenizer = get_moss_model_and_tokenizer(args)
    elif args.model_type == 'glm':
        model, tokenizer = get_glm_model_and_tokenizer(args)
    elif args.model_type == 'baichuan':
        model, tokenizer = get_baichuan_model_and_tokenizer(args)
    elif args.model_type == 'gptq':
        model, tokenizer = get_gptq_model_and_tokenizer(args)
    else:
        raise ValueError('invalid model_type, should be llama|moss|glm|baichuan|gptq')

    if args.do_batch and args.batch_size > 1:
        pass
    else:
        for d in tqdm(data):
            prompt = get_prompt_by_typ(d)
            s = time.time()
            res = inference(args, model, tokenizer, prompt, logits_processor=logits_processor)
            e = time.time()
            results.append(
                {'id': d['id'],
                 'prompt': prompt,
                 'answer': res
                }
            )
            logger.info('tokens:{}, inference_time:{}s'.format(len(res),e-s))


    if not os.path.exists(f'dataset/output/{args.model_name}/'):
        os.mkdir(f'dataset/output/{args.model_name}/')
    with open(f'dataset/output/{args.model_name}/{args.data_path.split("/")[-1].strip(".json")}_tem{args.temperature}_rep_{args.rep_penalty}.json', 'w') as f:
        f.write(json.dumps(results))
