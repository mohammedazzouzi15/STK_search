import json


def read_json(path):
    # """ """
    # if not path.endswith(".json"):
    #     raise UserWarning(f"Path {path} is not a json-path.")

    # with open(path, "r") as f:
    #    content = json.load(f)
    # return content
    """ """
    if path is None:
        return None  # or raise an error, depending on your requirements
    if not path.endswith(".json"):
        msg = f"Path {path} is not a json-path."
        raise UserWarning(msg)
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None  # or handle the absence of the file in an appropriate way



def update_json(path, data):
    """ """
    if not path.endswith(".json"):
        msg = f"Path {path} is not a json-path."
        raise UserWarning(msg)

    content = read_json(path)
    content.update(data)
    write_json(path, content)


def write_json(path, data):
    """ """
    if not path.endswith(".json"):
        msg = f"Path {path} is not a json-path."
        raise UserWarning(msg)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_value_json(path, key):
    """ """
    content = read_json(path)

    # if key in content.keys():
    #     return content[key]
    # else:
    #     return None
    if content is not None:
        if key in content:
            return content[key]
        else:
            return None
    else:
        return None
