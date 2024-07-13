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
        return parse_int(result)
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


def parse_int(input_str: str, default: int) -> int:
    """
    Tries to parse the input string into an integer. If the input string is
    missing (None) or malformed (cannot be converted to an integer),
    it returns the provided default integer.

    Parameters:
    input_str (str): The string to parse.
    default (int): The default value to return if parsing fails.

    Returns:
    int: The parsed integer or the default value.
    """
    # Check if the input string is None
    if input_str is None:
        return default

    try:
        # Attempt to convert the string to an integer
        return int(input_str.strip())
    except ValueError:
        # If conversion fails, return the default value
        return default
