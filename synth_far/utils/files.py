import json
import os

import yaml

JSON_EXTENSIONS = {"json"}
YAML_EXTENSIONS = {"yaml", "yml"}


def get_extension(path) -> str:
    return os.path.splitext(path)[1][1:]


def read_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, yaml.Loader)


def yaml_dump(obj, path, **kwargs):
    with open(path, "w") as f:
        yaml.dump(obj, f, **kwargs)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def json_dump(obj, path, **kwargs):
    with open(path, "w") as f:
        json.dump(obj, f, **kwargs)
