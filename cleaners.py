import re
import os
from typing import Optional, Set

COMMON_WORDLIST_PATHS = [
    '/usr/share/dict/words',
    '/usr/dict/words',
    '/usr/dict/web2',
    'wordlist.txt',
]


def remove_metadata(text: str) -> str:
    return re.sub(r'(?is)###\s*METADATA\s*###.*?###\s*END\s*METADATA\s*###\s*', '', text)


def remove_page_headers(text: str) -> str:
    # remove lines like "68 / Getting Back at James Joyce" at line starts
    return re.sub(r'(?m)^\s*\d+\s*/.*\n', '', text)


def normalize_unicode(text: str) -> str:
    repl = {
        '\u2014': ' - ',
        '\u2018': "'",
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
        '€': 'e',
        '\ufb01': 'fi',
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def load_wordlist(paths=COMMON_WORDLIST_PATHS) -> Set[str]:
    for p in paths:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                    return {w.strip().lower() for w in fh if w.strip()}
        except Exception:
            continue
    return set()


def clean_regex(text: str) -> str:
    text = remove_metadata(text)
    text = normalize_unicode(text)
    text = remove_page_headers(text)

    # fix hyphenation across line breaks (word-<newline>next -> wordnext)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # collapse multiple newlines into paragraph markers, then join single newlines to spaces
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'\n', ' ', text)

    # normalize whitespace and punctuation spacing
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([\.,;:\?!\)])', r'\1', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    return text


def clean_dict_hyphen(text: str, wordset: Optional[Set[str]] = None) -> str:
    """
    Like clean_regex but attempts to avoid joining two fragments if the joined word is not in a wordlist.
    If no wordlist is found, falls back to a conservative heuristic.
    """
    text = remove_metadata(text)
    text = normalize_unicode(text)
    text = remove_page_headers(text)

    if wordset is None:
        wordset = load_wordlist()

    def hyphen_fix(m):
        left, right = m.group(1), m.group(2)
        joined = (left + right).lower()
        if wordset and joined in wordset:
            return left + right
        # heuristic fallback: join if left+right shorter than 25 and both parts > 1 char
        if len(joined) <= 25 and len(left) > 1 and len(right) > 1:
            return left + right
        # otherwise keep a hyphen (we remove the newline but keep a hyphen)
        return left + '-' + right

    text = re.sub(r'(\w)-\s*\n\s*(\w+)', hyphen_fix, text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


if __name__ == '__main__':
    print('This module provides `clean_regex(text)` and `clean_dict_hyphen(text, wordset=None)`')
