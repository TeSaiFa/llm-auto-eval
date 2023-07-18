import torch
from transformers import AutoTokenizer, AutoModel
from src.models.modeling_baichuan import BaiChuanForCausalLM
from transformers.generation.utils import LogitsProcessorList

def get_baichuan_model_and_tokenizer(args):
    base_model = BaiChuanForCausalLM.from_pretrained(
                                           args.model_path,
                                           torch_dtype=torch.float16,
                                           trust_remote_code=True,
                                            # load_in_4bit=args.load_in_4bit,

    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = base_model
    model.eval()
    if not args.load_in_4bit:
        model.to("cuda:{}".format(args.device))
    return model, tokenizer

def baichuan_inference(args, model, tokenizer, prompt, logits_processor=None):
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