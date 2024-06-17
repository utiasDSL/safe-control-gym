def update_properties(obj, **kwargs):
    for k, v in kwargs.items():
        setattr(obj, k, v)
