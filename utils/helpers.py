import unidecode
import re


def strip_text(text: str) -> str:
    # remove html tags
    text = re.sub(r"<[^>]+>", "", text)

    # normalize chars
    text = unidecode.unidecode(text)

    # remove all non-alphanumeric characters, replace with a space
    text = re.sub(r"[^A-Za-z0-9]", " ", text)

    # convert to lowercase and strip extra whitespace
    clean_text = re.sub(r"\s+", " ", text).strip().lower()

    return clean_text
