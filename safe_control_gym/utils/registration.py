'''Environments and controllers registration.

Based on https://github.com/openai/gym/blob/master/gym/envs/registration.py
'''

import copy
import importlib
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources  # Try backported to PY<37 `importlib_resources`.

import yaml


def load(name):
    '''Loads a callable from a string with format `module_path:callable_name`.'''
    mod_name, attr_name = name.split(':')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class Spec():
    ''' A specification for a particular instance of the environment.'''

    def __init__(self, idx, entry_point=None, config_entry_point=None):
        '''Used in function 'register'.

        Args:
            idx (str): The official environment ID
            entry_point (cls, str): The Python entrypoint of the environment class.
            config_entry_point (str): The config to pass to the environment class.'''
        self.idx = idx
        self.entry_point = entry_point
        self.config_entry_point = config_entry_point

    def __repr__(self):
        '''Defines a class instance's string representation.'''
        return f'Spec({self.idx})'

    def get_config(self):
        '''Fetches config (as dict) for the registered class.'''
        if isinstance(self.config_entry_point, str):
            if self.config_entry_point.endswith('.yaml'):
                # Specified as file path.
                mod_name, config_name = self.config_entry_point.split(':')
                with pkg_resources.open_text(mod_name, config_name) as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                # Specified as 'module_path:config_dict_name'.
                config = load(self.config_entry_point)
        elif self.config_entry_point is None:
            # No default config.
            config = {}
        else:
            raise Exception(f'Config type {self.config_entry_point} is not supported.')
        return config

    def make(self, *args, **kwargs):
        '''Instantiates an instance of the registered class with appropriate kwargs.'''
        if self.entry_point is None:
            raise Exception(f'Attempting to make deprecated env {self.idx}.')
        if callable(self.entry_point):
            obj = self.entry_point(*args, **kwargs)
        else:
            cls = load(self.entry_point)
            obj = cls(*args, **kwargs)
        # Make the instance aware of which spec it came from.
        spec = copy.deepcopy(self)
        if hasattr(obj, 'spec'):
            obj.instance_spec = spec
        else:
            obj.spec = spec
        return obj


class Registry():
    '''Register a callable by ID.'''

    def __init__(self):
        self.specs = {}

    def make(self, path, *args, **kwargs):
        '''Create an instance of the registered callable by id `path`.'''
        spec = self.specs[path]
        obj = spec.make(*args, **kwargs)
        return obj

    def all(self):
        '''Returns all registered callables.'''
        return self.specs.values()

    def spec(self, path):
        '''Returns spec of the registered callable by id.'''
        if ':' in path:
            mod_name, _ = path.split(':')
            try:
                importlib.import_module(mod_name)
            except BaseException:
                raise Exception(f'''A module ({mod_name}) was specified for the environment but was not found,make sure the
                                package is installed with `pip install` before calling `gym.make()`''')
        else:
            idx = path
        try:
            return self.specs[idx]
        except BaseException:
            raise Exception('Key not found in registry.')

    def register(self, idx, **kwargs):
        '''Saves a reference to the callable with a id.'''
        if idx in self.specs:
            raise Exception(f'Cannot re-register id: {idx}')
        self.specs[idx] = Spec(idx, **kwargs)


def register(idx, **kwargs):
    '''Saves the callable to the global registry.'''
    return registry.register(idx, **kwargs)


def make(idx, *args, **kwargs):
    '''Creates an instance of the callable from global registry.'''
    return registry.make(idx, *args, **kwargs)


def spec(idx):
    '''Gets the spec of the callable from global registry.'''
    return registry.spec(idx)


def get_config(idx):
    '''Gets the config of the callable from global registry.'''
    return registry.spec(idx).get_config()


# Registry: global registry (singleton) used by functions `register`, `make`, `spec` and `get_config`.
registry = Registry()
