import asyncio
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from inspect_ai.model import (
    ModelAPI, modelapi, ModelOutput, ChatMessage, ChatMessageAssistant,
    GenerateConfig, ChatCompletionChoice, ToolInfo, ToolChoice
)

@modelapi("Reading2")
def create_steering_model():
    return QwenSteeringModel


class QwenSteeringModel(ModelAPI):
    def __init__(self,
        model_name: str = "Qwen/Qwen3-8B",
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **kwargs,
    ):

        self.model_path = kwargs.get("model_path", model_name)
        self.steering_coeff = float(kwargs.get("steering_coeff", 0.25))
        vectors_path = kwargs.get("vectors_path", None)

        super().__init__(model_name, base_url, api_key, api_key_vars, config)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Qwen has no pad_token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        #bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=None,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.hf_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Load steering vectors
        self.contrast_vectors = {}
        if vectors_path and Path(vectors_path).exists():
            self.contrast_vectors = torch.load(vectors_path)
        else:
            print("WARNING: No contrast vectors found, steering disabled.")

    # ------------------ STEERING ------------------

    def _generate_sync(self, inputs):
        hook_handles = []

        def create_steering_hook(vector):
            def hook(module, input, output):
                # Qwen block output: hidden_states only
                hidden_states = output
                v = vector.to(hidden_states.device, hidden_states.dtype)
                v = v.view(1, 1, -1)  # broadcast
                return hidden_states + self.steering_coeff * v
            return hook

        try:
            if self.contrast_vectors:
                for layer_idx, vector in self.contrast_vectors.items():
                    idx = int(layer_idx)
                    block = self.hf_model.model.layers[idx]
                    handle = block.register_forward_hook(
                        create_steering_hook(vector)
                    )
                    hook_handles.append(handle)

            with torch.no_grad():
                return self.hf_model.generate(**inputs,
                                              max_new_tokens=1024,
                                              do_sample=False,
                                              pad_token_id=self.tokenizer.pad_token_id)

        finally:
            for h in hook_handles:
                h.remove()

    # ------------------ INSPECT-AI ------------------

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
        **kwargs,
    ) -> ModelOutput:

        messages = [{"role": m.role, "content": m.content} for m in input]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.hf_model.device) for k, v in inputs.items()}

        output_ids = await asyncio.to_thread(self._generate_sync, inputs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=text),
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                )
            ],
        )
