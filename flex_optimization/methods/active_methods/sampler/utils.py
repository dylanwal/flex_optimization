def map_number(old_value, old_min, old_max, new_min, new_max) -> float:
    if old_value < old_min:
        return new_min
    if old_value > old_max:
        return new_max

    old_span = old_max - old_min
    new_span = new_max - new_min
    scaled_value = float(old_value - old_min) / float(old_span)
    return new_min + (scaled_value * new_span)
