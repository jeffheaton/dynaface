def get_bool(settings, key, default=False):
    """
    Retrieves a boolean value from the settings dictionary.

    Args:
        settings (dict): Dictionary containing settings.
        key (str): The key to look up in the dictionary.
        default (bool): The default value to return if the key is not found or the value is malformatted.

    Returns:
        bool: The value associated with the key, or the default value.
    """
    try:
        result = settings.get(key, default)
        if isinstance(result, bool):
            return result
        # Attempt to interpret the result as a boolean
        if str(result).strip().lower() in ["true", "yes", "1"]:
            return True
        elif str(result).strip().lower() in ["false", "no", "0"]:
            return False
        else:
            return default
    except:
        return default


def get_int(settings, key, default=1):
    """
    Retrieves an integer value from the settings dictionary.

    Args:
        settings (dict): Dictionary containing settings.
        key (str): The key to look up in the dictionary.
        default (int): The default value to return if the key is not found or the value is malformatted.

    Returns:
        int: The value associated with the key, or the default value.
    """
    try:
        result = settings.get(key, default)
        if isinstance(result, int):
            return result
        # Convert to integer if possible, else use default
        return int(str(result).strip())
    except (ValueError, TypeError):
        return default


def get_str(settings, key, default=""):
    """
    Retrieves a string value from the settings dictionary.

    Args:
        settings (dict): Dictionary containing settings.
        key (str): The key to look up in the dictionary.
        default (str): The default value to return if the key is not found or the value is malformatted.

    Returns:
        str: The value associated with the key, or the default value.
    """
    try:
        result = settings.get(key, default)
        if isinstance(result, str):
            return result
        return str(result)
    except:
        return default


def set_combo(cb, value):
    idx = cb.findText(value)
    if idx >= 0:
        cb.setCurrentIndex(idx)
    else:
        cb.setCurrentIndex(0)
