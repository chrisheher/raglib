import re
import os
from typing import Optional, Set

# --- Utility loaders and normalizers ---
COMMON_WORDLIST_PATHS = [
    '/usr/share/dict/words',
    '/usr/dict/words',
    '/usr/dict/web2',
    'wordlist.txt',
]

def load_wordlist(paths=COMMON_WORDLIST_PATHS) -> Set[str]:
    for p in paths:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                    return {w.strip().lower() for w in fh if w.strip()}
        except Exception:
            continue
    return set()


def normalize_unicode(text: str) -> str:
    repl = {
        '\u2014': ' - ',
        '\u2018': "'",
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2010': '-',
        '\u2011': '-',
        '\ufb01': 'fi',
        '€': 'e',
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def remove_metadata(text: str) -> str:
    return re.sub(r'(?is)###\s*METADATA\s*###.*?###\s*END\s*METADATA\s*###\s*', '', text)


def remove_page_headers(text: str) -> str:
    # remove lines like "68 / Getting Back at James Joyce" at starts
    text = re.sub(r'(?m)^\s*\d+\s*/[^\n]*\n', '', text)
    return text


def fix_hyphenation(text: str) -> str:
    # join hyphenated linebreaks: word-\n next -> wordnext
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return text


def collapse_newlines(text: str) -> str:
    # preserve paragraph breaks (2+ newlines) as \n\n, convert single newlines to spaces
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'\r', '', text)
    text = re.sub(r'\n', ' ', text)
    return text


def remove_control_chars_and_garble(text: str) -> str:
    # remove non-printable controls
    text = ''.join(ch for ch in text if ch == '\n' or (32 <= ord(ch) <= 126) or ord(ch) > 127)
    # remove isolated short garbage tokens like "yy" or repeated punctuation tokens
    text = re.sub(r'(?<=\s)[:;\-\.,]{2,}(?=\s)', ' ', text)
    # remove standalone repeated-letter tokens (e.g., "yy", "aaa") unlikely to be real words
    text = re.sub(r'(?m)(?<=\s)[A-Za-z]{2,}\b', lambda m: m.group(0) if len(set(m.group(0))) > 1 else ' ', text)
    return text

# --- Norvig-style spell corrector (pip-free) ---

_letters = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + (R[1:] if len(R) else '') for L, R in splits if R for c in _letters]
    inserts = [L + c + R for L, R in splits for c in _letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))


def known(words, wordset: Set[str]) -> Set[str]:
    return {w for w in words if w.lower() in wordset}


def correct_word(word: str, wordset: Set[str]) -> str:
    if not word or not word.isalpha():
        return word
    lower = word.lower()
    if lower in wordset:
        return word
    # candidates
    cands = known(edits1(lower), wordset) or known(edits2(lower), wordset) or None
    if cands:
        # pick shortest candidate (heuristic), preserve case
        best = min(cands, key=lambda w: (len(w), w))
        if word[0].isupper():
            return best.capitalize()
        return best
    return word

# --- Primary cleaner ---

def clean_advanced(text: str, wordset: Optional[Set[str]] = None, do_spell: bool = True) -> str:
    text = remove_metadata(text)
    text = normalize_unicode(text)
    text = remove_page_headers(text)
    text = fix_hyphenation(text)
    text = collapse_newlines(text)
    text = remove_control_chars_and_garble(text)

    # normalize spaces & punctuation spacing
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([\.,;:\?!])', r'\1', text)

    if do_spell:
        if wordset is None:
            wordset = load_wordlist()
        if wordset:
            # tokenize and correct
            def repl_token(m):
                token = m.group(0)
                corrected = correct_word(token, wordset)
                return corrected
            text = re.sub(r"\b[A-Za-z]{2,}\b", repl_token, text)
    return text


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Advanced pip-free text cleaner. Reads from file or stdin and writes to file or stdout.')
    parser.add_argument('-i', '--input', default='-', help="Input file path or '-' for stdin")
    parser.add_argument('-o', '--output', default='-', help="Output file path or '-' for stdout")
    parser.add_argument('--method', choices=['advanced', 'regex', 'dict'], default='advanced', help='Cleaning method')
    parser.add_argument('--no-spell', action='store_true', help='Disable spell correction')
    parser.add_argument('--encoding', default='utf-8', help='File encoding')

    args = parser.parse_args()

    # read input
    if args.input == '-':
        raw = sys.stdin.read()
    else:
        with open(args.input, 'r', encoding=args.encoding, errors='ignore') as fh:
            raw = fh.read()

    # load wordset if needed
    wordset = None
    if not args.no_spell and args.method in ('advanced', 'dict'):
        wordset = load_wordlist()

    # choose method
    out = raw
    if args.method == 'advanced':
        out = clean_advanced(raw, wordset=wordset, do_spell=not args.no_spell)
    else:
        try:
            import cleaners
        except Exception:
            cleaners = None
        if args.method == 'regex':
            if cleaners and hasattr(cleaners, 'clean_regex'):
                out = cleaners.clean_regex(raw)
            else:
                out = re.sub(r'(?is)###\s*METADATA\s*###.*?###\s*END\s*METADATA\s*###\s*', '', raw)
                out = fix_hyphenation(out)
                out = collapse_newlines(out)
                out = normalize_unicode(out)
        elif args.method == 'dict':
            if cleaners and hasattr(cleaners, 'clean_dict_hyphen'):
                out = cleaners.clean_dict_hyphen(raw, wordset=wordset)
            else:
                out = clean_advanced(raw, wordset=wordset, do_spell=not args.no_spell)

    # write output
    if args.output == '-':
        sys.stdout.write(out)
    else:
        with open(args.output, 'w', encoding=args.encoding) as fh:
            fh.write(out)
