from typing import Dict, Optional
from dataclasses import dataclass
from vllm import SamplingParams, LLM, ModelRegistry
from hf_olmo import *
from olmo_new import OlmoNewForCausalLM
import sys

@dataclass
class ModelConfig:
    """Configuration settings for model generation."""
    temperature: float = 0.8
    top_p: float = 0.95
    gpu_memory_utilization: float = 0.90
    default_prompt: str = "The capital of France is "

class ModelOutputGen:    
    def __init__(self, model_config: Optional[ModelConfig] = None):
        """Initialize the comparator with configuration."""
        self.config = model_config or ModelConfig()
        self._register_models()
    
    @staticmethod
    def _register_models() -> None:
        ModelRegistry.register_model("OlmoNewForCausalLM", OlmoNewForCausalLM)
    
    def _get_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )
    
    def _run_old_model(self, prompt: str) -> str:
        try:
            llm = LLM(model="allenai/OLMo-7B-hf")
            output = llm.generate([prompt], self._get_sampling_params())
            return output[0].outputs[0].text
        except Exception as e:
            raise RuntimeError(f"Error running old model: {str(e)}")
    
    def _run_new_model(self, model_path: str, prompt: str) -> str:
        try:
            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                gpu_memory_utilization=self.config.gpu_memory_utilization
            )
            output = llm.generate([prompt], sampling_params=self._get_sampling_params())
            return output[0].outputs[0].text
        except Exception as e:
            raise RuntimeError(f"Error running new model: {str(e)}")
    
    def run_models(
        self,
        model_path: str,
        prompt: Optional[str] = None,
        use_norm_reordering: str = "true"
    ) -> Dict[str, str]:
        """
        Generaate outputs between old and new VLLM models.
        
        Args:
            model_path: Path to the model checkpoint
            prompt: Input prompt for generation
            use_norm_reordering: Whether to use norm reordering ("true" or "false")
            
        Returns:
            Dictionary containing model outputs
        """
        if use_norm_reordering not in ["true", "false"]:
            raise ValueError("use_norm_reordering must be 'true' or 'false'")
            
        prompt = prompt or self.config.default_prompt
        outputs = {}
        
        if use_norm_reordering == "false":
            outputs["vllm"] = self._run_old_model(prompt)
        else:
            outputs["vllm"] = self._run_new_model(model_path, prompt)
            
        return outputs

def main():
    try:
        if len(sys.argv) < 2:
            raise ValueError("Model path argument is required")
            
        vllm_result_generator = ModelOutputGen()
        
        if len(sys.argv) == 2:
            results = vllm_result_generator.run_models(sys.argv[1])
        else:
            results = vllm_result_generator.run_models(
                model_path=sys.argv[1],
                use_norm_reordering=sys.argv[2]
            )
            
        print("\nResults:", results)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()