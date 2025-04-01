import diffusers
import torch
from PIL import Image
from munch import Munch
import random
import json
import glob
import tqdm
import os
import click


from synth_far.diffusion_pipelines import LPWStableDiffusionControlNetPipeline

with open("./dataset/wildcards.json") as f:
    wildcards = json.load(f)

controlnet_images = [Image.open(im) for im in glob.glob("./dataset/pose_images/pose_*.png")]


def sample_age():
    return random.random() * 85

def sample_gender(age: int):
    is_female = random.random() > 0.5
    
    if is_female:
        if age <= 5:
            gender = "infant girl, "
        elif age <= 10:
            gender = "child girl"
        elif age <= 15:
            gender = "teen age girl"
        elif age <= 25:
            gender = "college age girl"
        elif age <= 30:
            gender = "young adult woman"
        elif age <= 45:
            gender = "adult woman"
        elif age <= 60:
            gender = "middle age woman"
        else:
            gender = "grandmother"
    else:
        if age <= 5:
            gender = "infant boy, "
        elif age <= 10:
            gender = "child boy"
        elif age <= 15:
            gender = "teen age boy"
        elif age <= 25:
            gender = "college age boy"
        elif age <= 30:
            gender = "young adult man"
        elif age <= 45:
            gender = "adult man"
        elif age <= 60:
            gender = "middle age man"
        else:
            gender = "grandfather"
    return is_female, gender
            
def sample_ethnicity():
    id2ethnicity = {
        0: "white",
        1: "black",
        2: "asian",
        3: "indian",
        4: "other"
    }
    r = random.randint(0, 4)
    return r, id2ethnicity[r]

def make_random_options():
    age = sample_age()
    is_female, gender = sample_gender(age)
    ethnicity_id, ethnicity = sample_ethnicity()
    
    hair_color = random.choice(wildcards['hair_colors'])
    if is_female:
        hair_style = random.choice(wildcards['female_hairstyle'])
    else:
        hair_style = random.choice(wildcards['male_hairstyle'])
    hair_length = random.choice(wildcards['hair_length'])
    suffix = random.choice(wildcards["professions"]) + ", " + ", ".join(random.sample(
        wildcards["attributes"] , random.randint(1, 4)
    ))
    country = random.choice(list(wildcards['countries'][ethnicity].keys()))
    if is_female:
        name = random.choice(wildcards['countries'][ethnicity][country]['female_names'])
    else:
        name = random.choice(wildcards['countries'][ethnicity][country]['male_names'])
    
    if ethnicity == "black":
        country += ", black skin"
        
    options = Munch(
        age=int(age),
        gender=gender,
        ethnic=country,
        name=name,
        hair_length=hair_length,
        hair_style=hair_style,
        hair_color=hair_color,
        suffix=suffix,
    )
    labels = {
        "age": int(age),
        "female": is_female,
        "ethnicity": ethnicity_id,
        "hair_length": hair_length,
        "hair_color": hair_color,
    }
    return options, labels

def make_face_prompt(options):
    prompt = f"""
close-up photorealistic, raw photo,amateur photo,face,
({options.age} y.o) ({options.gender}) (from {options.ethnic}) named {options.name},
{options.hair_length} {options.hair_style} {options.hair_color} hair,
""".replace(
        "\n", ""
    ).replace(
        ":.", ":0."
    ) + options.suffix

    negative_prompt = """
(hands), (3d, render, cgi, doll, painting, fake, 3d modeling:1.4),
(worst quality, low quality:1.4), monochrome, child, deformed, malformed,
deformed face, bad teeth, bad hands, bad fingers, bad eyes, long body, blurry,
duplicated, cloned, duplicate body parts, disfigured, extra limbs, fused fingers,
extra fingers, twisted, distorted, malformed hands, mutated hands and fingers,
conjoined, missing limbs, bad anatomy, bad proportions, logo, watermark, text,
copyright, signature, lowres, mutated, mutilated, artifacts, gross, ugly, malformed genital
""".replace(
        "\n", ""
    )
    return prompt, negative_prompt, random.choice(controlnet_images)

@click.command()
@click.option(
    "--dest-folder", type=click.Path(file_okay=False, dir_okay=True)
)
@click.option(
    "--total-images", type=click.IntRange(0),
)
@click.option(
    "--num-images-per-folder", type=click.IntRange(0),
)
def main(
    dest_folder: str,
    total_images: int,
    num_images_per_folder: int, 
):
    controlnet = diffusers.ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16,
    ).to('cuda')
    sd = LPWStableDiffusionControlNetPipeline.from_single_file(
        "./pretrained_models/humans.safetensors",
        torch_dtype=torch.float16,
        controlnet=controlnet
    )
    sd.to('cuda')
    sd.set_progress_bar_config(disable=True)

    sd.scheduler = diffusers.DPMSolverMultistepScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear",
        use_karras_sigmas=True
    )

    os.makedirs(dest_folder, exist_ok=True)

    num_folders = total_images // num_images_per_folder

    current_image = len(glob.glob(
        os.path.join(dest_folder, "*", "*.jpg")
    ))

    for folder_num in tqdm.trange(0, num_folders + 1, desc="Folders"):
        try:
            if (folder_num + 1) * num_images_per_folder < current_image:
                continue
            current_folder = os.path.join(
                dest_folder, f"{folder_num * num_images_per_folder}"
            )
            os.makedirs(current_folder, exist_ok=True)
            labels_path = os.path.join(current_folder, "labels.json")
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as f:
                    labels = json.load(f)
            else:
                labels = {}

            for image_id in tqdm.trange(0, num_images_per_folder, desc="Images", leave=False):
                if image_id + num_images_per_folder * folder_num < current_image:
                    continue
                if image_id + num_images_per_folder * folder_num > total_images:
                    break
                try:
                    options, l = make_random_options()
                    p, np, control_image = make_face_prompt(options) 
                    image = sd(
                        image=control_image,
                        prompt=p,
                        negative_prompt=np,
                        guidance_scale=7.5,
                        num_inference_steps=15,
                        height=512, width=512,
                        controlnet_conditioning_scale=0.8,
                    ).images[0]
                    l['prompt'] = p
                    l['negative_prompt'] = np
                    labels[int(image_id + num_images_per_folder * folder_num)] = l
                    image.save(os.path.join(current_folder, f"{int(image_id + num_images_per_folder * folder_num)}.jpg"))
                except Exception as e:
                    print(e)
        except KeyboardInterrupt as e:
            print("Keyboard Interrupt, attempting gracefull shutdown")
            with open(labels_path, 'w') as f:
                json.dump(labels, f, indent=2)
            raise KeyboardInterrupt()
        
        with open(labels_path, 'w') as f:
                json.dump(labels, f, indent=2)

            
            


if __name__ == "__main__":
    main()
        