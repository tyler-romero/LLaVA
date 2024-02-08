import os
from io import BytesIO

import requests
import torch
from cog import BaseModel, BasePredictor, Input, Path
from PIL import Image
from transformers.generation import GreedySearchDecoderOnlyOutput

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/weights"

# See options here: https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md
ACTIVE_MODEL = "liuhaotian/llava-v1.6-mistral-7b"


def get_model_name_from_path(model_path):
    if "ShareGPT4V-7B" in model_path:
        return "llava-v1.5-7b"

    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class Output(BaseModel):
    """
    The output of the Predictor. This class is REQUIRED to be named "Output".
    """

    output: str
    top_tokens: list[str]
    top_token_ids: list[int]
    token_logprobs: list[float]


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        disable_torch_init()
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = load_pretrained_model(
            model_path=ACTIVE_MODEL,
            model_name=get_model_name_from_path(ACTIVE_MODEL),
            model_base=None,
            load_8bit=False,
            load_4bit=False,
        )

        eos_punctuation = [".", "!", "?"]
        self.punctuation_ids = []
        for p in eos_punctuation:
            cur_keyword_ids = self.tokenizer.convert_tokens_to_ids(p)
            self.punctuation_ids.append(cur_keyword_ids)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt to use for text generation"),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic",
            default=0.01,
            ge=0.0,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            default=1024,
            ge=0,
        ),
        logprobs: int = Input(
            description="Number of logprobs to return", ge=0, le=10, default=0
        ),
    ) -> Output:
        """Run a single prediction on the model"""

        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        image_data = load_image(str(image))
        image_tensor = (
            self.image_processor.preprocess(image_data, return_tensors="pt")[
                "pixel_values"
            ]
            .half()
            .cuda()
        )

        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(
            conv.roles[1], None
        )  # Set the assistant up for the response
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            outputs: GreedySearchDecoderOnlyOutput = (
                self.model.generate(  # BS, SEQ_LEN, VOCAB_SIZE
                    input_ids,
                    images=image_tensor,
                    do_sample=True if temperature > 0.01 else False,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    output_scores=True,  # return prediction logits
                    return_dict_in_generate=True,
                )
            )

        decoded_outputs = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        ).strip()  # [BS=1, SEQ_LEN] -> [SEQ_LEN]

        generated_sequence = outputs.sequences[0]  # [BS=1, SEQ_LEN] -> [SEQ_LEN]
        scores = outputs.scores  # [SEQ_LEN][BS=1, VOCAB_SIZE]

        # Remove stopping keywords and punctuation from end of the generated sequence
        while (
            generated_sequence[-1]
            in [self.tokenizer.eos_token_id] + self.punctuation_ids
        ):
            generated_sequence = generated_sequence[:-1]
            scores = scores[:-1]  # [SHORTED_SEQ_LEN][BS=1, VOCAB_SIZE]

        # Extract logprobs for last token in the seq. [SHORTED_SEQ_LEN][BS, VOCAB_SIZE] -> [VOCAB_SIZE]
        last_token_logits = scores[-1][0, :]
        last_token_logprobs = torch.log_softmax(last_token_logits, dim=0)

        top_k_results = torch.topk(last_token_logprobs, k=logprobs, dim=0)
        top_k_logprobs = top_k_results.values.cpu().tolist()
        top_k_token_ids = top_k_results.indices.cpu().tolist()
        top_k_tokens: list[str] = self.tokenizer.batch_decode(top_k_token_ids)

        decoded_outputs = self.tokenizer.decode(
            generated_sequence, skip_special_tokens=True
        ).strip()

        return Output(
            output=decoded_outputs,
            top_tokens=top_k_tokens,
            top_token_ids=top_k_token_ids,
            token_logprobs=top_k_logprobs,
        )


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image
