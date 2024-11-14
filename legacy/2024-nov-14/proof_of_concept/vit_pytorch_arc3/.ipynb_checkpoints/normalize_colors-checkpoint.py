def normalize_color(color: int) -> int:
    if color < 0:
        return 0
    if color > 12:
        return 255
    return color * 255 // 12
