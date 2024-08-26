import json


class School:
    def __init__(self, params_path):
        config = json.load(open(params_path, 'r'))
