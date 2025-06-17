from ominicontrol import generate_image
import os
import click

USE_ZERO_GPU = os.environ.get("USE_ZERO_GPU", "0") == "1"


styles = [
    "Studio Ghibli",
    "Irasutoya Illustration",
    "The Simpsons",
    "Snoopy",
]
inference_modes = ["High Quality", "Fast"]
image_ratios = ["Auto", "Square(1:1)", "Portrait(2:3)", "Landscape(3:2)"]
seed = 42
image_guidance = 1.5
step = 12


@click.command()
@click.option("--style", type=click.Choice(styles), default="Studio Ghibli")
@click.option("--original_image", type=click.Path(exists=True))
@click.option("--inference_mode", type=click.Choice(inference_modes), default="Fast")
@click.option("--image_guidance", type=float, default=image_guidance)
@click.option("--image_ratio", type=click.Choice(image_ratios), default="Square(1:1)")
@click.option("--seed", type=int, default=seed)
@click.option("--steps", type=int, default=step)
def infer(
    style,
    original_image,
    inference_mode,
    image_guidance,
    image_ratio,
    use_random_seed,
    seed,
    steps,
):
    print(
        f"Style: {style}, Inference Mode: {inference_mode}, Image Guidance: {image_guidance}, Image Ratio: {image_ratio}, Use Random Seed: {use_random_seed}, Seed: {seed}"
    )
    result_image = generate_image(
        image=original_image,
        style=style,
        inference_mode=inference_mode,
        image_guidance=image_guidance,
        image_ratio=image_ratio,
        use_random_seed=use_random_seed,
        seed=seed,
        steps=steps,
    )
    return result_image
