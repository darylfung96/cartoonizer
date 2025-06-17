import torch
from diffusers.pipelines import FluxPipeline
from OminiControl.src.flux.condition import Condition
from PIL import Image
import random

from OminiControl.src.flux.generate import generate, seed_everything

print("Loading model...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

pipe.unload_lora_weights()

pipe.load_lora_weights(
    "Yuanshi/OminiControlArt",
    weight_name=f"v0/ghibli.safetensors",
    adapter_name="ghibli",
)
pipe.load_lora_weights(
    "Yuanshi/OminiControlArt",
    weight_name=f"v0/irasutoya.safetensors",
    adapter_name="irasutoya",
)
pipe.load_lora_weights(
    "Yuanshi/OminiControlArt",
    weight_name=f"v0/simpsons.safetensors",
    adapter_name="simpsons",
)
pipe.load_lora_weights(
    "Yuanshi/OminiControlArt",
    weight_name=f"v0/snoopy.safetensors",
    adapter_name="snoopy",
)


def generate_image(
    image,
    style,
    inference_mode,
    image_guidance,
    image_ratio,
    steps,
    use_random_seed,
    seed,
):  
    # Prepare Condition
    def resize(img, factor=16):
        w, h = img.size
        new_w, new_h = w // factor * factor, h // factor * factor
        padding_w, padding_h = (w - new_w) // 2, (h - new_h) // 2
        img = img.crop((padding_w, padding_h, new_w + padding_w, new_h + padding_h))
        return img

    # Set Adapter
    activate_adapter_name = {
        "Studio Ghibli": "ghibli",
        "Irasutoya Illustration": "irasutoya",
        "The Simpsons": "simpsons",
        "Snoopy": "snoopy",
    }[style]
    pipe.set_adapters(activate_adapter_name)

    factor = 512 / max(image.size)
    image = resize(
        image.resize(
            (int(image.size[0] * factor), int(image.size[1] * factor)),
            Image.LANCZOS,
        )
    )
    delta = -image.size[0] // 16
    condition = Condition(
        "subject",
        # activate_adapter_name,
        image,
        position_delta=(0, delta),
    )

    # Prepare seed
    if use_random_seed:
        seed = random.randint(0, 2**32 - 1)
    seed_everything(seed)

    # Image guidance scale
    image_guidance = 1.0 if inference_mode == "Fast" else image_guidance

    # Output size
    if image_ratio == "Auto":
        r = image.size[0] / image.size[1]
        ratio = min([0.67, 1, 1.5], key=lambda x: abs(x - r))
    else:
        ratio = {
            "Square(1:1)": 1,
            "Portrait(2:3)": 0.67,
            "Landscape(3:2)": 1.5,
        }[image_ratio]
    width, height = {
        0.67: (640, 960),
        1: (640, 640),
        1.5: (960, 640),
    }[ratio]

    print(
        f"Image Ratio: {image_ratio}, Inference Mode: {inference_mode}, Image Guidance: {image_guidance}, Seed: {seed}, Steps: {steps}, Size: {width}x{height}"
    )
    # Generate
    result_img = generate(
        pipe,
        prompt="",
        conditions=[condition],
        num_inference_steps=steps,
        width=width,
        height=height,
        image_guidance_scale=image_guidance,
        default_lora=True,
        max_sequence_length=32,
    ).images[0]

    return result_img


def vote_feedback(
    log_id,
    feedback,
):
    log_data = {
        "log_id": log_id,
        "feedback": feedback,
    }
    log_data = {k: str(v) for k, v in log_data.items()}

