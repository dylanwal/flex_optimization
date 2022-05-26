

def get_dimensionality(args) -> int:
    """determine dimensionality"""
    if isinstance(args, (list, tuple)):
        return len(args)
    else:  # np.ndarray
        if len(args.shape) == 1:
            return args.size
        else:
            return args.shape[1]
