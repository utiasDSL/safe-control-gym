"""Example Google style docstrings.

Reference to https://github.com/openai/gym/blob/master/gym/envs/registration.py

"""
import copy
import yaml
import importlib
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources  # Try backported to PY<37 `importlib_resources`.


def load(name):
    """Loads a callable from a string with format `module_path:callable_name`."""
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class Spec():
    """ A specification for a particular instance of the environment."""

    def __init__(self, id, entry_point=None, config_entry_point=None):
        """Used in function "register".

        Args:
            id (str): The official environment ID
            entry_point (cls, str): The Python entrypoint of the environment class.
            config_entry_point (str): The config to pass to the environment class.

        """
        self.id = id
        self.entry_point = entry_point
        self.config_entry_point = config_entry_point

    def __repr__(self):
        """Defines a class instance's string representation."""
        return "Spec({})".format(self.id)

    def get_config(self):
        """Fetches config (as dict) for the registered class."""
        if isinstance(self.config_entry_point, str):
            if self.config_entry_point.endswith(".yaml"):
                # Specified as file path.
                mod_name, config_name = self.config_entry_point.split(":")
                with pkg_resources.open_text(mod_name, config_name) as f:
                    config = yaml.load(f)
            else:
                # Specified as "module_path:config_dict_name".
                config = load(self.config_entry_point)
        elif self.config_entry_point is None:
            # no default config (provided as overwrites)
            config = {}
        else:
            raise Exception("Config type {} is not supported.".format(
                self.config_entry_point))
        return config

    def make(self, *args, **kwargs):
        """Instantiates an instance of the registered class with appropriate kwargs."""
        if self.entry_point is None:
            raise Exception('Attempting to make deprecated env {}.'.format(
                self.id))
        if callable(self.entry_point):
            obj = self.entry_point(*args, **kwargs)
        else:
            cls = load(self.entry_point)
            obj = cls(*args, **kwargs)
        # Make the instance aware of which spec it came from.
        spec = copy.deepcopy(self)
        if hasattr(obj, "spec"):
            obj.instance_spec = spec
        else:
            obj.spec = spec
        return obj


class Registry():
    """Register a callable by ID."""

    def __init__(self):
        self.specs = {}

    def make(self, path, *args, **kwargs):
        """Create an instance of the registered callable by id `path`."""
        spec = self.specs[path]
        obj = spec.make(*args, **kwargs)
        return obj

    def all(self):
        """Returns all registered callables."""
        return self.specs.values()

    def spec(self, path):
        """Returns spec of the registered callable by id."""
        if ':' in path:
            mod_name, id = path.split(':')
            try:
                importlib.import_module(mod_name)
            except:
                raise Exception(
                    '''A module ({}) was specified for the environment but was not
                                found, make sure the package is installed with `pip install`
                                before calling `gym.make()`'''.format(mod_name))
        else:
            id = path

        try:
            return self.specs[id]
        except:
            raise Exception("Key not found in registry.")

    def register(self, id, **kwargs):
        """Saves a reference to the callable with a id."""
        if id in self.specs:
            raise Exception('Cannot re-register id: {}'.format(id))
        self.specs[id] = Spec(id, **kwargs)


def register(id, **kwargs):
    """Saves the callable to the global registry."""
    return registry.register(id, **kwargs)


def make(id, *args, **kwargs):
    """Creates an instance of the callable from global registry."""
    return registry.make(id, *args, **kwargs)


def spec(id):
    """Gets the spec of the callable from global registry."""
    return registry.spec(id)


def get_config(id):
    """Gets the config of the callable from global registry."""
    return registry.spec(id).get_config()


# Registry: global registry (must be a singleton).
# used by functions `register`, `make`, `spec` and `get_config`.
registry = Registry()
