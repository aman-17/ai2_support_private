from hf_olmo import *
s = SamplingParams(temperature=0.0)
llm = LLM(model=path, trust_remote_code=True, gpu_memory_utilization=0.90)

set_random_seed(0)
vllm_out = llm.generate([prompt], sampling_params=s)
outputs["vllm"] = vllm_out[0].outputs[0].text