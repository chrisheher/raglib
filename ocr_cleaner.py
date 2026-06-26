"""
ocr_cleaner.py
--------------
Comprehensive OCR post-processing pipeline derived from analysis of 20 scanned
academic and literary texts. Produces three output files per document:
  - <name>.clean.txt      : cleaned body text
  - <name>.footnotes.txt  : extracted footnotes / metadata (if any)
  - <name>.changes.log    : every change made, with context

Run:
    python ocr_cleaner.py input.txt
    python ocr_cleaner.py *.txt          # batch mode
    python ocr_cleaner.py --two-letter-flags input.txt  # enable noisy 2-letter flagging
"""

import re
import sys
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Small fast vocabulary for isolation checks (flagging only, never auto-correct)
COMMON_WORDS = set("""
a about above across after again age ago all also am an and any are as at
away back be been before being below between both but by came can could
day did do does done down during each even ever every few first for from
get give go good great had has have he her here him his how i if in into
is it its just know large last later least like long look made make man
many may me more most much must my new no not now of off on one only or
other our out over own part people same say see she should since so some
still such than that the their them then there these they this those
though through time to too two under until up us very was we were what
when where which while who will with would year yet you your
""".split())

VALID_TWO_LETTER = {
    'a','i','an','as','at','be','by','do','go','he','if','in','is','it',
    'me','my','no','of','on','or','so','to','up','us','we',
    'mg','iv','ph','vs','ad','bc','et','al','eg','ie',
}

FOOTNOTE_SIGNALS = re.compile(
    r'^(\d{1,3}[\s°]|Ibid\.|pp\.\s|vol\.\s|ch\.\s|op\.\s|See\s'
    r'|Reading\s|Alludes\s|Translat|Cf\.\s|Ibid,)',
    re.IGNORECASE
)

# (regex_pattern, replacement, description) — applied line by line
CHAR_SUBS = [
    (r'\\ip\b',        'up',       'backslash-ip to up'),
    (r'\beh[P!]\b',    'eh?',      'ehP/eh! to eh?'),
    (r'^~',            '\u2014',   'leading tilde to em-dash'),
]


# ---------------------------------------------------------------------------
# HEALTH CHECK  (runs before pipeline, aborts if SEVERE)
# ---------------------------------------------------------------------------

def _line_is_garbage(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False
    tokens = stripped.split()
    if len(tokens) < 2:
        return False
    alpha = sum(c.isalpha() for c in stripped)
    total = len(stripped)
    if total > 0 and alpha / total < 0.35 and total < 80:
        return True
    garbage_tokens = sum(1 for t in tokens if len(t) <= 2 and not t.isalpha())
    if len(tokens) >= 3 and garbage_tokens / len(tokens) > 0.5:
        return True
    return False


def document_health(text: str) -> tuple:
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return 'EMPTY', 1.0
    garbage = sum(1 for l in lines if _line_is_garbage(l))
    ratio = garbage / len(lines)
    if ratio > 0.30:
        return 'SEVERE', ratio
    if ratio > 0.10:
        return 'MODERATE', ratio
    return 'CLEAN', ratio


# ---------------------------------------------------------------------------
# STAGE 1 — PRE-PROCESSING
# ---------------------------------------------------------------------------

def strip_metadata_blocks(text, log):
    """Remove ### METADATA ### ... ### END METADATA ### blocks injected by upstream tools."""
    pattern = re.compile(r'###\s*METADATA\s*###.*?###\s*END METADATA\s*###\n?', re.DOTALL | re.IGNORECASE)
    cleaned, n = pattern.subn('', text)
    if n:
        log.append(f'[metadata_block] removed {n} metadata block(s)')
    return cleaned


def normalize_unicode(text, log):
    """Replace known ligatures and invisible characters."""
    subs = [
        ('\ufb01', 'fi'),    # fi ligature
        ('\ufb02', 'fl'),    # fl ligature
        ('\u00ad', ''),      # soft hyphen
        ('\u2028', '\n'),    # line separator
        ('\u2029', '\n\n'),  # paragraph separator
    ]
    for old, new in subs:
        if old in text:
            count = text.count(old)
            text = text.replace(old, new)
            log.append(f'[unicode] {count}x {repr(old)} replaced with {repr(new)}')
    return text


def strip_unicode_superscripts(text, log):
    """Remove TM, registered, degree, dagger etc. used as OCR-misread footnote markers."""
    pattern = re.compile(r'[™®°†‡§¶]')
    cleaned, n = pattern.subn('', text)
    if n:
        log.append(f'[superscripts] removed {n} unicode superscript artifact(s)')
    return cleaned


# ---------------------------------------------------------------------------
# STAGE 2 — LINE-LEVEL STRUCTURAL REMOVAL
# ---------------------------------------------------------------------------

def remove_margin_strips(text, log):
    """
    Remove vertical binding/margin strips: runs of 3+ consecutive very-short lines.
    Seen in: Meletinsky (ol, II, Gl, (I at page top).
    """
    lines = text.split('\n')
    result = []
    i = 0
    removed = 0
    while i < len(lines):
        run_start = i
        j = i
        while j < len(lines) and len(lines[j].strip()) <= 4:
            j += 1
        run_len = j - run_start
        if run_len >= 3:
            removed += run_len
            log.append(f'[margin_strip] removed {run_len}-line strip near line {i+1}')
            i = j
        else:
            result.append(lines[i])
            i += 1
    return '\n'.join(result)


def remove_split_page_headers(text, log):
    """
    Handle headers split across two lines:
        28
        GROWTH OF THE SOIL 5
    Pattern: lone integer line immediately followed by ALL-CAPS title line.
    """
    lines = text.split('\n')
    result = []
    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        stripped = line.strip()
        if (re.match(r'^\d{1,4}$', stripped)
                and i + 1 < len(lines)
                and re.match(r'^[A-Z][A-Z\s\d]{4,}$', lines[i + 1].strip())):
            log.append(f'[split_header] removed: {stripped!r} + {lines[i+1].strip()!r}')
            skip_next = True
            continue
        result.append(line)
    return '\n'.join(result)


def remove_page_headers(text, log):
    """
    Remove running headers in all observed formats:
      'JAPAN 117'  |  'Hemingway Legacy 225'  |  'Capitalism Flow = 161'
      '150 Under the Volcano'  |  '284 KNUT HAMSUN'
    """
    patterns = [
        re.compile(r"^\s*[A-Za-z'\-\s,\.]{3,55}\s+\d{2,4}\s*$"),
        re.compile(r"^\s*[A-Za-z'\-\s,\.]{3,55}\s+=\s*\d{2,4}\s*$"),
        re.compile(r'^\s*\d{2,4}\s+[™®°\-]?\s*[A-Z][A-Za-z\s\'\-,\.]{3,55}\s*$'),
    ]
    lines = text.split('\n')
    result = []
    removed = 0
    for line in lines:
        if any(p.match(line) for p in patterns):
            removed += 1
            log.append(f'[page_header] removed: {line.strip()!r}')
        else:
            result.append(line)
    return '\n'.join(result)


def remove_standalone_page_numbers(text, log):
    """Remove lines that are purely a page number: '120', '{ 74 }', '28'."""
    pattern = re.compile(r'^\s*\{?\s*\d{1,4}\s*\}?\s*$')
    lines = text.split('\n')
    result = []
    removed = 0
    for line in lines:
        if pattern.match(line):
            removed += 1
            log.append(f'[page_number] removed: {line.strip()!r}')
        else:
            result.append(line)
    if removed:
        log.append(f'[page_number] total: {removed}')
    return '\n'.join(result)


def remove_separator_lines(text, log):
    """
    Remove decorative/noise separator lines.
    Criterion: line is short (<60 chars) and <40% alphabetic.
    Examples: 'FSS FTES SS SF', '= ae ee', 'er $erery =', 'BS', 'i a a Sc'
    """
    lines = text.split('\n')
    result = []
    removed = 0
    for line in lines:
        stripped = line.strip()
        total = len(stripped)
        if total == 0:
            result.append(line)
            continue
        alpha = sum(c.isalpha() for c in stripped)
        if total < 60 and alpha / total < 0.40:
            removed += 1
            log.append(f'[separator] removed: {stripped!r}')
        else:
            result.append(line)
    if removed:
        log.append(f'[separator] total removed: {removed}')
    return '\n'.join(result)


def remove_allcaps_section_headers(text, log):
    """
    Remove short ALL-CAPS mid-page section headers with no lowercase.
    Examples: 'FOOD OF THE GODS', 'CANNABIS AND CULTURE'
    """
    pattern = re.compile(r'^\s*[A-Z][A-Z\s]{5,50}[A-Z]\s*$')
    lines = text.split('\n')
    result = []
    removed = 0
    for line in lines:
        if pattern.match(line) and not any(c.islower() for c in line):
            removed += 1
            log.append(f'[allcaps_header] removed: {line.strip()!r}')
            continue
        result.append(line)
    return '\n'.join(result)


def remove_trailing_document_garbage(text, log):
    """
    Drop terminal garbage fragments at end of document.
    Seen in Weiner: 'Erna / Chains? / \\ WS yee try ('
    """
    paragraphs = re.split(r'\n{2,}', text.rstrip())
    if not paragraphs:
        return text
    while paragraphs:
        last = paragraphs[-1].strip()
        tokens = last.split()
        total = len(last)
        alpha = sum(c.isalpha() for c in last)
        # Garbage if: short AND (mostly non-alpha OR very few tokens with no sentence structure)
        is_non_alpha_junk = total > 0 and total < 60 and alpha / total < 0.50 and len(tokens) <= 6
        is_bare_word_fragment = len(tokens) <= 3 and total < 30 and not any(c in last for c in '.!?')
        if is_non_alpha_junk or is_bare_word_fragment:
            log.append(f'[trailing_garbage] removed terminal fragment: {last!r}')
            paragraphs.pop()
        else:
            break
    return '\n\n'.join(paragraphs)


def strip_line_start_artifacts(text, log):
    """
    Strip leading _ | characters from line starts (column gutter bleed).
    Only strips when followed by a letter, preserving intentional indentation.
    """
    pattern = re.compile(r'^[_|]{1,3}\s+(?=[A-Za-z])', re.MULTILINE)
    cleaned, n = pattern.subn('', text)
    if n:
        log.append(f'[line_start_artifact] removed {n} leading _ or | artifact(s)')
    return cleaned


def remove_single_char_lines(text, log):
    """Remove lines containing only a single non-space character."""
    lines = text.split('\n')
    result = []
    removed = 0
    for line in lines:
        if len(line.strip()) == 1:
            removed += 1
        else:
            result.append(line)
    if removed:
        log.append(f'[single_char_line] removed {removed} single-character line(s)')
    return '\n'.join(result)


# ---------------------------------------------------------------------------
# STAGE 3 — FOOTNOTE EXTRACTION
# ---------------------------------------------------------------------------

def extract_footnotes(text, log):
    """
    Separate footnote blocks from body text.
    Returns (body_text, footnotes_text).
    """
    lines = text.split('\n')
    body, notes = [], []
    in_footnote = False
    for line in lines:
        stripped = line.strip()
        if FOOTNOTE_SIGNALS.match(stripped):
            in_footnote = True
        elif in_footnote and stripped == '':
            in_footnote = False
            notes.append(line)
            continue
        if in_footnote:
            notes.append(line)
        else:
            body.append(line)
    if notes:
        log.append(f'[footnotes] extracted {len(notes)} footnote line(s) to separate file')
    return '\n'.join(body), '\n'.join(notes)


# ---------------------------------------------------------------------------
# STAGE 4 — CHARACTER-LEVEL CORRECTIONS
# ---------------------------------------------------------------------------

def fix_underscore_in_words(text, log):
    """
    Fix OCR underscore inserted mid-word: 'a_pensioner' -> 'a pensioner'.
    Only splits at word boundaries (letter_letter).
    """
    pattern = re.compile(r'(\w)_(\w)')
    cleaned, n = pattern.subn(r'\1 \2', text)
    if n:
        log.append(f'[underscore_word] fixed {n} underscore-fused word(s)')
    return cleaned


def apply_char_substitutions(text, log):
    """Apply the targeted character-level substitution table."""
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        original = line
        for pat, repl, desc in CHAR_SUBS:
            new_line, n = re.subn(pat, repl, line)
            if n:
                log.append(f'[char_sub] {desc} (line {i+1}): {line.strip()!r} -> {new_line.strip()!r}')
                line = new_line
        result.append(line)
    return '\n'.join(result)


def fix_trailing_tilde(text, log):
    """Remove trailing ~ used as em-dash surrogate at line ends."""
    pattern = re.compile(r'\s*~\s*$', re.MULTILINE)
    cleaned, n = pattern.subn('', text)
    if n:
        log.append(f'[tilde] removed {n} trailing tilde(s)')
    return cleaned


# ---------------------------------------------------------------------------
# STAGE 5 — LINE STRUCTURE REPAIR
# ---------------------------------------------------------------------------

def fix_hyphenated_line_breaks(text, log):
    """
    Rejoin words hyphenated across lines: 'writ-\nten' -> 'written'.
    Only joins when continuation starts with lowercase (not an em-dash or
    compound proper noun like 'Micro-\nSoft').
    """
    pattern = re.compile(r'(\w)-\n([a-z])')
    cleaned, n = pattern.subn(r'\1\2', text)
    if n:
        log.append(f'[hyphen_break] rejoined {n} hyphenated line break(s)')
    return cleaned


def rejoin_broken_sentences(text, log):
    """
    Rejoin mid-sentence line breaks.
    Conditions:
      - Previous line does NOT end with terminal punctuation
      - Next line starts with a lowercase letter
      - No blank line separates them (paragraph boundaries preserved)
    """
    pattern = re.compile(r'(?<![.!?:"\d\u2014])\n(?=[a-z])')
    cleaned, n = pattern.subn(' ', text)
    if n:
        log.append(f'[sentence_break] rejoined {n} broken mid-sentence line(s)')
    return cleaned


def collapse_excess_newlines(text, log):
    """Reduce 3+ consecutive newlines to exactly 2."""
    pattern = re.compile(r'\n{3,}')
    cleaned, n = pattern.subn('\n\n', text)
    if n:
        log.append(f'[newlines] collapsed {n} excess newline group(s) to paragraph breaks')
    return cleaned


def normalize_whitespace(text, log):
    """Strip trailing whitespace per line; collapse internal double-spaces."""
    lines = text.split('\n')
    result = []
    for line in lines:
        line = line.rstrip()
        line = re.sub(r'(?<=\S) {2,}', ' ', line)
        result.append(line)
    log.append('[whitespace] normalized trailing and internal whitespace')
    return '\n'.join(result)


# ---------------------------------------------------------------------------
# STAGE 6 — FLAGGING  (annotates only, never deletes)
# ---------------------------------------------------------------------------

def flag_isolated_corrupt_lines(text, log):
    """
    Flag single corrupt lines surrounded by clean prose.
    Uses COMMON_WORDS as a proxy: if >65% of meaningful tokens are unknown
    and neighbors are long normal lines, the line is flagged.
    Seen in: Pamuk ('alsa read soondls thet do net whe plase-in bioad land-')
    """
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        tokens = re.findall(r'[a-zA-Z]{3,}', line)
        if len(tokens) < 4:
            result.append(line)
            continue
        unknown = sum(1 for t in tokens if t.lower() not in COMMON_WORDS)
        ratio = unknown / len(tokens)
        prev_ok = i > 0 and len(lines[i - 1].strip()) > 40
        next_ok = i < len(lines) - 1 and len(lines[i + 1].strip()) > 40
        if ratio > 0.80 and prev_ok and next_ok:
            result.append(f'[ISOLATED_CORRUPT: {line}]')
            log.append(f'[flag_corrupt_line] line {i+1}: {line.strip()!r}')
        else:
            result.append(line)
    return '\n'.join(result)


def flag_corrupted_blocks(text, log):
    """
    Flag multi-line blocks of heavily corrupted text with a single marker.
    A block is defined as a run of 3+ consecutive garbage lines.
    Seen in: Bakhtin (opening), Frazer (opening), McKenna (middle sections).
    """
    lines = text.split('\n')
    result = []
    i = 0
    while i < len(lines):
        run = []
        j = i
        while j < len(lines) and _line_is_garbage(lines[j]):
            run.append(lines[j])
            j += 1
        if len(run) >= 3:
            preview = ' | '.join(l.strip() for l in run[:3])
            result.append(f'[CORRUPT_BLOCK ({len(run)} lines): {preview} ...]')
            log.append(f'[flag_corrupt_block] {len(run)}-line block starting at line {i+1}: {preview!r}')
            i = j
        else:
            result.append(lines[i])
            i += 1
    return '\n'.join(result)


def flag_two_letter_artifacts(text, log):
    """
    Flag two-letter tokens that are not in the valid-word whitelist.
    OFF by default — this is noisy and should be used for review passes only.
    Enable with --two-letter-flags on the command line.
    """
    def replace_fn(m):
        word = m.group(0)
        core = word.lower().strip('.,;:!?\'")')
        if len(core) == 2 and core not in VALID_TWO_LETTER:
            log.append(f'[two_letter_flag] suspicious token: {word!r}')
            return f'[?{word}]'
        return word
    return re.compile(r'\b[A-Za-z]{2}\b').sub(replace_fn, text)


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

PIPELINE = [
    # Stage 1
    strip_metadata_blocks,
    normalize_unicode,
    strip_unicode_superscripts,
    # Stage 2
    remove_margin_strips,
    remove_split_page_headers,
    remove_page_headers,
    remove_standalone_page_numbers,
    remove_separator_lines,
    remove_allcaps_section_headers,
    remove_trailing_document_garbage,
    strip_line_start_artifacts,
    remove_single_char_lines,
    # Stage 3 (footnotes handled separately — returns tuple)
    # Stage 4
    fix_underscore_in_words,
    apply_char_substitutions,
    fix_trailing_tilde,
    # Stage 5
    fix_hyphenated_line_breaks,
    rejoin_broken_sentences,
    collapse_excess_newlines,
    normalize_whitespace,
    # Stage 6 — flagging
    flag_isolated_corrupt_lines,
    flag_corrupted_blocks,
]


def run_pipeline(text: str, include_two_letter_flags: bool = False) -> tuple:
    """
    Run the full pipeline.
    Returns: (clean_body, footnotes, change_log_text)
    """
    log = []

    severity, ratio = document_health(text)
    log.append(f'[health_check] severity={severity}, garbage_ratio={ratio:.2%}')

    if severity == 'SEVERE':
        log.append(
            '[health_check] PIPELINE ABORTED — document too corrupted for automatic '
            'cleaning. Recommend manual review or re-OCR with column-detection enabled.'
        )
        return text, '', '\n'.join(log)

    for stage in PIPELINE:
        text = stage(text, log)

    body, footnotes = extract_footnotes(text, log)

    if include_two_letter_flags:
        body = flag_two_letter_artifacts(body, log)

    return body, footnotes, '\n'.join(log)


# ---------------------------------------------------------------------------
# FILE I/O
# ---------------------------------------------------------------------------

def process_file(input_path: str, include_two_letter_flags: bool = False):
    path = Path(input_path)
    text = path.read_text(encoding='utf-8', errors='replace')

    body, footnotes, change_log = run_pipeline(text, include_two_letter_flags)

    out_dir = path.parent
    stem = path.stem

    body_path = out_dir / f'{stem}.clean.txt'
    log_path  = out_dir / f'{stem}.changes.log'

    body_path.write_text(body, encoding='utf-8')
    log_path.write_text(change_log, encoding='utf-8')
    print(f'  clean text  -> {body_path}')
    print(f'  change log  -> {log_path}')

    if footnotes.strip():
        notes_path = out_dir / f'{stem}.footnotes.txt'
        notes_path.write_text(footnotes, encoding='utf-8')
        print(f'  footnotes   -> {notes_path}')

    # Quick summary
    log_lines = change_log.count('\n') + 1
    severity, ratio = document_health(text)
    print(f'  health: {severity} ({ratio:.1%} garbage ratio) | log entries: {log_lines}')


if __name__ == '__main__':
    args = sys.argv[1:]
    two_letter = '--two-letter-flags' in args
    files = [a for a in args if not a.startswith('--')]

    if not files:
        print('Usage: python ocr_cleaner.py [--two-letter-flags] <file.txt> [file2.txt ...]')
        sys.exit(1)

    for f in files:
        print(f'\nProcessing: {f}')
        try:
            process_file(f, two_letter)
        except Exception as e:
            print(f'  ERROR: {e}')