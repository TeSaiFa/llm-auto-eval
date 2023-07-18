import argparse

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from transformers.generation.utils import LogitsProcessorList

def get_llama_model_and_tokenizer(args):
    base_model = LlamaForCausalLM.from_pretrained(args.model_path,
                                                  torch_dtype=torch.float16,
                                                  # load_in_4bit=args.load_in_4bit,
                                                  )
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size != tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    model = base_model
    model.eval()
    if not args.load_in_4bit:
        model.to("cuda:{}".format(args.device))
    return model, tokenizer

def llama_inference(args, model, tokenizer, prompt, logits_processor=None):
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

    # if args.do_batch and args.batch_size > 1:
    #     batch_size = args.batch_size
    #     num = len(data)//batch_size + 1
    #     for i in range(num):
    #         batch_data = data[i*batch_size:min((i+1)*batch_size,len(data))]
    #         batch_prompts = []
    #         for d in batch_data:
    #             prompt = get_prompt_by_typ(d)
    #             batch_prompts.append(prompt)
    #         input_ids = tokenizer.encode(batch_prompts, return_tensors="pt", padding_side='left').to("cuda:{}".format(args.device))
    #         with torch.no_grad():
    #             generated_ids = model.generate(
    #                 input_ids,
    #                 do_sample=args.do_sample,
    #                 min_length=args.min_length,
    #                 max_length=args.max_length,
    #                 top_p=args.top_p,
    #                 top_k=args.top_k,
    #                 temperature=args.temperature,
    #                 return_dict_in_generate=True,
    #                 repetition_penalty=args.rep_penalty
    #             )
    #

    # else:



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('--data_path', type=str, help='dataset path')
    parser.add_argument('--min_length', type=int, default=10, help='The minimum length of the sequence to be generated.')
    parser.add_argument('--max_length', type=int, default=2048, help='The maximum length of the sequence to be generated.')

    parser.add_argument('--top_p',
                        type=float,
                        default=0.95,
                        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.')
    parser.add_argument('--top_k',
                        type=int,
                        default=40,)
    parser.add_argument('--temperature', type=float, default=0.8, help='The value used to module the next token probabilities.')
    parser.add_argument('--device', type=int, default=-1, help='The device used to load the model when using safetensors. Default device is "cpu" or specify, 0,1,2,3,... for GPU device.')
    parser.add_argument('--args.repetition_penalty', type=float, default=1.0)

    args = parser.parse_args()


    model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.eval()

    model.to("cuda:{}".format(args.device))
    tokenizer = LlamaTokenizer.from_pretrained(args.model)


    import json
    data = json.load(open(args.data_path), 'r')
    for d in data:
        input_ids = tokenizer.encode(d, return_tensors="pt").to("cuda:{}".format(args.device))
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                do_sample=False,
                min_length=args.min_length,
                max_length=args.max_length,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                return_dict_in_generate=True,
                repetition_penalty=args.repetition_penalty
            )
    # print(tokenizer.decode([el.item() for el in generated_ids[0]]))