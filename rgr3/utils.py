def fmt(value):
    """Округление для красивого вывода."""
    if isinstance(value, str):
        return value
    return f"{value:.4f}"
