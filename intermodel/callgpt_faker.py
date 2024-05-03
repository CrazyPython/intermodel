from intermodel.hf_token import get_hf_tokenizer
from typing import Union, List, Optional


def fake_local(
    model,
    prompt=None,
    max_tokens=None,
    stop=None,
    frequency_penalty: Union[float, int] = 0,
    presence_penalty: Union[float, int] = 0,
    num_completions: Optional[int] = None,
    top_k=None,
    repetition_penalty: Union[float, int] = 1,
    tfs=1,
    user_id=None,
    logit_bias=None,
    vendor=None,
    vendor_config=None,
    **kwargs,
):
    # local import for performance
    import tiktoken
    import random
    import string
    import time

    get_hf_tokenizer(model)
    try:
        enc = tiktoken.encoding_for_model(model)
        encode = lambda s: enc.encode(s, allowed_special="all")
        get_random_token = lambda: random.choice(valid_tokens).decode("utf-8")
    except KeyError:
        enc = get_hf_tokenizer(model)
        encode = lambda s: enc.encode(s, add_special_tokens=True)
        valid_tokens = list(enc.get_vocab().keys())
        get_random_token = lambda: random.choice(valid_tokens)
    else:
        valid_tokens = list(enc._mergeable_ranks.keys())
    completions = []
    if num_completions is None:
        num_completions = 1
    for i in range(num_completions):
        n_tokens = random.randint(0, max_tokens)
        s = ""
        # try to add tokens until we reach n_tokens
        while len(encode(s)) < n_tokens:
            try:
                new_token = get_random_token()
            except UnicodeDecodeError:
                continue
            if stop is not None and new_token in stop:
                break
            if logit_bias is not None and logit_bias.get(new_token, 0) <= -100:
                continue
            else:
                s += new_token
        # trim until we are below it
        while len(encode(s)) > n_tokens:
            s = s[:-1]
        completions.append(s)

    completion_tokens = sum([len(encode(item)) for item in completions])

    return {
        "prompt": {
            "text": prompt if prompt is not None else get_default_prompt(model, vendor)
        },
        "completions": [
            {
                "text": completion,
                "finish_reason": {
                    "reason": (
                        "length"
                        if len(encode(completion))
                        == max_tokens
                        else "stop"
                    )
                },
            }
            for completion in completions
        ],
        "model": model,
        "id": f"fake-{[random.choice(string.ascii_letters) for i in range(30)]}",
        "created": time.time(),
        "usage": {
            # openai always returns prompt_tokens: 1 minimum, even if prompt=""
            "prompt_tokens": max(len(encode(prompt)), 1),
            # if the completion is empty, the value will be missing
            "completion_tokens": completion_tokens,
            "charged_tokens": len(encode(prompt))
            + completion_tokens,
            "vendor": vendor,
        },
    }


"""
Return the default prompt used when prompt=None
"""


def get_default_prompt(model, vendor):
    if vendor == "openai":
        return "<|endoftext|>"
    else:
        return ""


def fake_remote_openai(model):
    import tiktoken

    enc: tiktoken.Encoding = tiktoken.encoding_for_model(model)
    if enc.name == "cl100k_base":
        return "text-embedding-ada-002"
    elif enc.name == "r50k_base":
        return "ada"
    elif enc.name.startswith("p50k"):
        return "text-davinci-edit-001"
