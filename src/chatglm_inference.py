import torch
from transformers import AutoTokenizer, AutoModel
from transformers.generation.utils import LogitsProcessorList

def get_glm_model_and_tokenizer(args):
    base_model = AutoModel.from_pretrained(args.model_path,
                                           torch_dtype=torch.float16,
                                           trust_remote_code=True,
                                           # load_in_4bit=args.load_in_4bit,

                                           )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = base_model
    if args.use_fastllm:
        from fastllm_pytools import llm
        model = llm.from_hf(model, tokenizer, dtype="float16")

    model.eval()
    if not args.load_in_4bit:
        model.to("cuda:{}".format(args.device))
    return model, tokenizer

def glm_inference(args, model, tokenizer, prompt, logits_processor=None):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:{}".format(args.device))
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=args.do_sample,
            max_length=args.max_length,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            return_dict_in_generate=True,
            repetition_penalty=args.rep_penalty,
            logits_processor=logits_processor if logits_processor else LogitsProcessorList(),
            pad_token_id=tokenizer.pad_token_id
        )
    res = tokenizer.decode(generated_ids.sequences[0][len(inputs.input_ids[0]):])
    return res