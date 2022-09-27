from cleaner.GbcProductAppraisalsStringUtils import GbcProductAppraisalsStringUtils
from preload import PreLoad
from spellchecker import SpellChecker


class SpanishSpellChecker:
    """
    is_readable, ratio = spell_checker_service.is_spanish(txt_data)
    """

    def __init__(self):
        self.spell = SpellChecker(language='es', distance=2)

    def is_spanish(self, doc) -> (bool, float):
        if not doc:
            return False, 0

        work_list = self.tokenizer(doc)

        # find those words that may be misspelled
        misspelled = self.spell.unknown(work_list)
        spelled = self.spell.known(work_list)

        ratio = len(spelled) / len(misspelled)
        return len(spelled) >= len(misspelled), ratio

    @staticmethod
    def tokenizer(sentence):
        token_list = list()
        clean_sentence = GbcProductAppraisalsStringUtils.speller_clean(sentence)
        nlp = PreLoad.getInstance().getSpanishModel()
        doc = nlp.analyze(clean_sentence)
        for token in doc:
            if (
                    token.pos_ == r'PROPN' or
                    token.pos_ == r'NOUN' or
                    token.pos_ == r'ADJ' or
                    token.pos_ == r'ADV' or
                    token.pos_ == r'AUX' or
                    token.pos_ == r'INTJ' or
                    token.pos_ == r'VERB' or
                    token.pos_ == r'ADP' or
                    token.pos_ == r'DET'
            ):
                token_list.append(token.text)

        return token_list
