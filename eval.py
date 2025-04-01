from synth_far.datasets import FairFaceDataset, UTKDataset
from synth_far.modeling.age_timm import AGETimmModel, AGETimmModelAgeCls
from synth_far.utils.files import read_json
import os.path as osp
import torch
from tqdm import tqdm
from sklearn.metrics._regression import mean_absolute_error
from sklearn.metrics._classification import accuracy_score
import numpy as np
import click


def load_from_checkpoint(checkpoint_path):
    config = read_json(osp.join(checkpoint_path, "config.json"))
    model_config = config['model']['model']
    model_type = model_config.pop('type')
    model_config.pop("weights", None)
    if model_type.endswith("Cls"):
        model = AGETimmModelAgeCls(**model_config)
    else:
        model = AGETimmModel(**model_config)

    state_dict = torch.load(osp.join(checkpoint_path, "model.pth"))

    model.load_state_dict(state_dict)
    model.eval();
    model.cuda();
    return model, config['val_dataloader']['dataset']['transforms']


@torch.no_grad()
def get_preds_utk(model, transform_path):
    dataset = UTKDataset(
        "./dataset/utk-split.json", 'test', transform_path
    )
    A = []
    G = []
    E = []
    A_hat = []
    G_hat = []
    E_hat = []
    
    for sample in tqdm(dataset, leave=False):
        A.append(sample[1]['age'].item())
        G.append(sample[1]['gender'].item())
        E.append(sample[1]['ethnicity'].item())
        model_inp = sample[0].cuda().unsqueeze(0)
        model_out = model(model_inp)
        if not isinstance(model, AGETimmModelAgeCls):
            A_hat.append(model_out['age'].item())
        G_hat.append((model_out['gender'].sigmoid() > 0.5).int().item())
        E_hat.append(model_out['ethnicity'].argmax().int().item())
        
    return {
        "target": {
            "age": A,
            "gender": G,
            "ethnicity": E,
        },
        "pred": {
            "age": A_hat,
            "gender": G_hat,
            "ethnicity": E_hat,
        }
    }


def age2group(age):
    return torch.bucketize(age, torch.tensor([3, 10, 20, 30, 40, 50, 60, 70, 200]).to(age.device)).int()

@torch.no_grad()
def get_preds_ff(model, transform_path):
    dataset = FairFaceDataset(
        "", 'test', transform_path
    )
    A = []
    G = []
    E = []
    A_hat = []
    G_hat = []
    E_hat = []
    
    for sample in tqdm(dataset, leave='False'):
        A.append(sample[1]['age'].item())
        G.append(sample[1]['gender'].item())
        E.append(sample[1]['ethnicity'].item())
        model_inp = sample[0].cuda().unsqueeze(0)
        model_out = model(model_inp)
        if not isinstance(model, AGETimmModelAgeCls):
            A_hat.append(age2group(model_out['age']).item())
        else:
            A_hat.append(model_out['age'].argmax().int().item())
        G_hat.append((model_out['gender'].sigmoid() > 0.5).int().item())
        E_hat.append(model_out['ethnicity'].argmax().int().item())
        
    return {
        "target": {
            "age": A,
            "gender": G,
            "ethnicity": E,
        },
        "pred": {
            "age": A_hat,
            "gender": G_hat,
            "ethnicity": E_hat,
        }
    }

@click.command()
@click.option("--checkpoint", type=str, required=True)
def main(checkpoint):
    model, transform_path = load_from_checkpoint(checkpoint)
    r = get_preds_utk(model, transform_path)
    print("***********UTK***********")
    if not isinstance(model, AGETimmModelAgeCls):
        print(f"Age MAE: {mean_absolute_error(np.clip(np.array(r['target']['age']), 0, 150),np.clip(np.array(r['pred']['age']), 0, 150)):.2f}")
    print(f"Gender accuracy: {accuracy_score(r['target']['gender'], r['pred']['gender']):.2f}")
    print(f"Ethnicity accuracy: {accuracy_score(r['target']['ethnicity'], r['pred']['ethnicity']):.2f}")

    r = get_preds_ff(model, transform_path)
    print("***********FairFace***********")
    print(f"Age accuracy: {accuracy_score(r['target']['age'], r['pred']['age']):.2f}")
    print(f"Gender accuracy: {accuracy_score(r['target']['gender'], r['pred']['gender']):.2f}")
    print(f"Ethnicity accuracy: {accuracy_score(r['target']['ethnicity'], r['pred']['ethnicity']):.2f}")


if __name__ == "__main__":
    main()