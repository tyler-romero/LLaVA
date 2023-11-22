import os
from io import BytesIO

import requests
import torch
from cog import BaseModel, BasePredictor, Input, Path
from PIL import Image
from transformers.generation import GreedySearchDecoderOnlyOutput

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/weights"

ACTIVE_MODEL = "Lin-Chen/ShareGPT4V-7B"

MODEL_TO_ARCH = {
    "Lin-Chen/ShareGPT4V-7B": "llava-v1.5-7b",  # https://arxiv.org/pdf/2311.12793.pdf
    "liuhaotian/llava-v1.5-13b": "llava-v1.5-13b",  # https://arxiv.org/abs/2310.03744
}


def download_json(url: str, dest: Path) -> None:
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")


class Output(BaseModel):
    """
    The output of the Predictor. This class is REQUIRED to be named "Output".
    """

    output: str
    retrieve_scores_for_tokens: list[str]
    token_probabilities: list[float]


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
            model_name=MODEL_TO_ARCH[ACTIVE_MODEL],
            model_base=None,
            load_8bit=False,
            load_4bit=False,
        )

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
        retrieve_scores_for_tokens: str = Input(
            description="Comma separated list of tokens to return scores for",
            default="",
        ),
    ) -> Output:
        """Run a single prediction on the model"""

        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        # Get the token_ids for the tokens we want to retrieve scores for
        retrieve_scores_for_token_ids = {}
        if len(retrieve_scores_for_tokens) > 0:
            retrieve_scores_for_tokens = retrieve_scores_for_tokens.split(",")
            for token in retrieve_scores_for_tokens:
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                print(token_ids)
                if len(token_ids) != 1:
                    raise ValueError(
                        f"Token {token} must be a single token"
                    )  # TODO: abstract to sequences of tokens
                token_id = token_ids[0]
                retrieve_scores_for_token_ids[token_id] = token

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
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
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
                    stopping_criteria=[stopping_criteria],
                    return_dict_in_generate=True,
                )
            )

        generated_sequence = outputs.sequences[
            0, input_ids.shape[1] :
        ]  # [BS=1, SEQ_LEN] -> [NEW_SEQ_LEN]
        scores = outputs.scores  # [NEW_SEQ_LEN][BS, VOCAB_SIZE]

        # Remove stopping keywords from end of the generated sequence
        keyword_ids = [
            keyword_id.to(generated_sequence.device)
            for keyword_id in stopping_criteria.keyword_ids
        ]
        while generated_sequence[-1] in keyword_ids:
            generated_sequence = generated_sequence[
                :-1
            ]  # remove last token if it is a keyword
            scores = scores[:-1]

        last_token_scores = scores[-1][
            0, :
        ]  # [NEW_SEQ_LEN][BS, VOCAB_SIZE] -> [VOCAB_SIZE]

        # Compute the probability scores for the specified tokens and '__other__' category
        token_probabilities = {}
        sum_of_specified_token_probs = torch.tensor(
            0.0, device=last_token_scores.device
        )
        for token_id, token in retrieve_scores_for_token_ids.items():
            token_probability = (
                torch.softmax(last_token_scores, dim=0)[token_id].cpu().item()
            )
            token_probabilities[token] = token_probability
            sum_of_specified_token_probs += token_probability

        # Calculate the total probability of all other tokens by subtraction
        token_probabilities["__other__"] = 1 - sum_of_specified_token_probs.cpu().item()

        decoded_outputs = self.tokenizer.decode(
            generated_sequence, skip_special_tokens=True
        ).strip()
        conv.messages[-1][-1] = decoded_outputs

        if decoded_outputs.endswith(stop_str):
            decoded_outputs = decoded_outputs[: -len(stop_str)].strip()

        return Output(
            output=decoded_outputs,
            retrieve_scores_for_tokens=list(token_probabilities.keys()),
            token_probabilities=list(token_probabilities.values()),
        )


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image
