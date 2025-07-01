def extract_number_manual(key: str) -> int | None:
    i = len(key) - 1
    while i >= 0 and key[i].isdigit():
        i -= 1
    number_part = key[i + 1 :]
    return int(number_part) if number_part else None
