import re

import unicodedata


class ExtractorService:
    get_section_part_extension = [
        re.compile(r"(c\)\.?-?\s|c\.-\s|c\.\)\s)",
                   re.IGNORECASE | re.MULTILINE | re.DOTALL),
        re.compile(r"(3\)\.?-?\s|3\.-\s|3\.\)\s)",
                   re.IGNORECASE | re.MULTILINE | re.DOTALL),
        re.compile(r"(e\)\.?-?\s|e\.-\s|e\.\)\s)",
                   re.IGNORECASE | re.MULTILINE | re.DOTALL),
        re.compile(r"(4\)\.?-?\s|4\.-\s|4\.\)\s)",
                   re.IGNORECASE | re.MULTILINE | re.DOTALL),
        re.compile(r"(e\)\.?-?\s|e\.-\s|e\.\)\s)",
                   re.IGNORECASE | re.MULTILINE | re.DOTALL),
        re.compile(r"(5\)\.?-?\s|5\.-\s|5\.\)\s)",
                   re.IGNORECASE | re.MULTILINE | re.DOTALL),
        re.compile(r"(f\)\.?-?\s|f\.-\s|f\.\)\s)",
                   re.IGNORECASE | re.MULTILINE | re.DOTALL),
        re.compile(r"(6\)\.?-?\s|6\.-\s|6\.\)\s)",
                   re.IGNORECASE | re.MULTILINE | re.DOTALL),
    ]

    def __init__(self):
        pass

    @staticmethod
    def simplify(text):
        clean_text = text
        try:
            accents = ('COMBINING ACUTE ACCENT', 'COMBINING GRAVE ACCENT', 'COMBINING TILDE')
            accents = set(map(unicodedata.lookup, accents))
            chars = [c for c in unicodedata.normalize('NFD', text) if c not in accents]
            if chars:
                clean_text = unicodedata.normalize('NFC', ''.join(chars))
        except NameError:
            pass

        return str(clean_text)

    @staticmethod
    def clean_doc(text):
        if text is None:
            return ''

        new_ = re.sub(r'\n', ' ', text)
        new_ = re.sub(r'\n\n', r'\n', new_)
        new_ = re.sub(r'\\n', ' ', new_)
        new_ = re.sub(r'\t', ' ', new_)

        # page number
        new_ = re.sub(r'(\[\[\[.{1,3}\]\]\])', '', new_)
        new_ = re.sub(r'\- F.?o.?l.?i.?o \d{1,2} \-', '', new_)
        new_ = re.sub(r"F.?o.?l.?i.?o.{0,2}-.{0,2}[0-9]{1,2}.{0,2}-", '', new_)

        new_ = re.sub(r'(\d{5})-(\W*)(\w{3,20}?)\W', r'\1 \2', new_)

        new_ = re.sub(r'--', r'-', new_)
        new_ = re.sub(r'=', ' ', new_)
        new_ = re.sub(r':', ' ', new_)
        new_ = re.sub(r'\( ', r'(', new_)
        new_ = re.sub(r' \)', r')', new_)
        new_ = re.sub(r'"', ' ', new_)
        new_ = re.sub(r'_', ' ', new_)
        new_ = re.sub(r'\'', ' ', new_)
        new_ = re.sub(r'/', ' ', new_)

        # fix #54312 para eliminar los guiones pero no en nif o nie ejm 3-2 y 3-1D queda 3 2 y 3 1D
        new_ = re.sub(r'(\s[a-z0-9]{1,3})-([a-z0-9]{1,3}\s)', r'\1 \2', new_)

        new_ = re.sub(r'\s{2,1000}', " ", new_)

        new_ = re.sub(r'\((\d{5})\)', r'\1', new_)  # inside curly brackets
        new_ = re.sub(r'\((?!CP)[.\w\s\-%]{1,100}\)', r' ', new_)  # inside curly brackets but not CP for postal code
        new_ = re.sub(r"(\w{2})-\s(\w{2})", r'\1\2', new_)  # join new line words

        return new_.strip()

    @staticmethod
    def compact_doc(text):
        if text is None:
            return ''

        new_ = re.sub(r'\n\n', r'\n', text)
        new_ = re.sub(r'\n', ' ', new_)
        new_ = re.sub(r'\\n', ' ', new_)
        new_ = re.sub(r'\t', ' ', new_)

        return new_.strip()
