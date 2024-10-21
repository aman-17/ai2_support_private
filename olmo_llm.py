"""
Intermediate checkpoints missing for OLMo-1B. LINK: https://huggingface.co/allenai/OLMo-1B/discussions/11
"""
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast

olmo = OLMoForCausalLM.from_pretrained("allenai/OLMo-1B", revision="step733000-tokens3074B")
tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-1B")
message = ["Language modeling is"]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])