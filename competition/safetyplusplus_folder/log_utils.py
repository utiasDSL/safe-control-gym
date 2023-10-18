#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'MICROYU'

import json

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False

def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) 
                    for k,v in obj.items()}

        if isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v) 
                        for k,v in obj.__dict__.items() if not str(k).startswith('_')}
            return {'__dict__': obj_dict}

        return str(obj)

def save_config(exp_name, config, log_dir):
    config_json = convert_json(config)
    if exp_name is not None:
        config_json['exp_name'] = exp_name
    output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
    print(colorize('Saving config:', color='cyan', bold=True))
    print(colorize(output, color='cyan', bold=True))
    with open(log_dir + "/config.json", 'w') as out:
        out.write(output)