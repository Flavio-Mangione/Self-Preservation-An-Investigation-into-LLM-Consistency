import asyncio
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from inspect_ai.model import (
    ModelAPI, modelapi, ModelOutput, ChatMessage, ChatMessageAssistant,
    GenerateConfig, ChatCompletionChoice, ToolInfo, ToolChoice
)

# Register the model with the framework
@modelapi("Reading")
def create_steering_model():
    return LlamaSteeringModel

class LlamaSteeringModel(ModelAPI):
    def __init__(self,
        model_name: str = "llama2-7b",
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **kwargs,):
        
        self.model_path = kwargs.get("model_path", None)
        self.steering_coeff = float(kwargs.get("steering_coeff", 0.25))
        vectors_path = kwargs.get("vectors_path", None)  

        super().__init__(model_name, base_url, api_key, api_key_vars, config)

        # Load Tokenizer and Model
        print(f"Loading tokenizer from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        print(f"Loading model from: {self.model_path}")

        bnb_config = BitsAndBytesConfig(load_in_8bit = True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config = bnb_config,
            device_map = "auto",
            torch_dtype = torch.float16)
        
        print(f"Model loaded on device: {next(self.hf_model.parameters()).device}")

        # Load Contrast Vectors
        self.contrast_vectors = {}
        if vectors_path and Path(vectors_path).exists():
            self.contrast_vectors = torch.load(vectors_path, weights_only=True)
            print(f"Loaded contrast vectors from {vectors_path}. Layers: {list(self.contrast_vectors.keys())}")
            print(f"Steering coefficient: {self.steering_coeff}")
        else:
            print(f"WARNING: No contrast vectors found at {vectors_path}, steering disabled.")

    def _generate_sync(self, inputs):

        """
        Performs synchronous generation while applying steering hooks.
        """

        hook_handles = []

        def create_steering_hook(vector):
            def hook(module, input, output):
                # Llama output is a tuple (hidden_states, past_key_values, ...)
                # We take only the hidden states
                if isinstance(output, tuple):
                    hidden_states = output[0] # like output.logits
                else:
                    hidden_states = output

                v = vector.to(hidden_states.device).to(hidden_states.dtype)
                steered_activation = hidden_states + (self.steering_coeff * v)
                
                if isinstance(output, tuple):
                    return (steered_activation,) + output[1:]
                else:
                    return steered_activation
                
            return hook

        try:
            # Register hooks on specified layers
            if self.contrast_vectors:
                for layer_idx, vector in self.contrast_vectors.items():
                    try:
                        idx = int(layer_idx) 
                        # Access the specific Llama 2 layer
                        target_layer = self.hf_model.model.layers[idx]
                        
                        # Register the hook
                        handle = target_layer.register_forward_hook(create_steering_hook(vector))
                        hook_handles.append(handle)
                    except Exception as e:
                        print(f"Error registering hook on layer {layer_idx}: {e}")

            # Run generation (hooks will now modify activations)
            with torch.no_grad():
                return self.hf_model.generate(**inputs)

        finally:
            # Remove ALL hooks to clean the model for the next call
            for handle in hook_handles:
                handle.remove()

    async def generate(self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
        **kwargs,) -> ModelOutput:
        
        prompt_messages = [{"role": m.role, "content": m.content} for m in input]
        prompt = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(next(self.hf_model.parameters()).device) for k, v in inputs.items()}

        # Execution on a separate thread to avoid blocking asyncio
        output_ids = await asyncio.to_thread(self._generate_sync, inputs)

        # Decoding Part
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Defining ModelOutput
        return ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(content=response_text),
                finish_reason="stop",
                index=0,
                logprobs=None
            )]
        )