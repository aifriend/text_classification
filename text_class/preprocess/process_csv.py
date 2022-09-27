from warnings import simplefilter

import pandas as pd
import stanza
from texthero.preprocessing import remove_digits, remove_diacritics, remove_whitespace, remove_brackets, \
    remove_html_tags, \
    remove_urls

simplefilter(action='ignore', category=FutureWarning)

nlp = stanza.Pipeline(lang='es', processors='tokenize')


def sentence_splitter(doc_list):
    doc_sentences = list()
    for idx, doc in enumerate(doc_list):
        print(f"{idx}* ", end='')
        nlp_doc = nlp(doc)
        for sentence in nlp_doc.sentences:
            doc_sentences.append(sentence.text)
    return doc_sentences


def remove_regex(_input: pd.Series) -> pd.Series:
    return _input.str.replace(r'[!"#)($%&*+-/<=>?@[\]^_`{|}~]', " ").str.split().str.join(" ")


def main():
    df = pd.read_csv(
        "reg_data.csv",
        header=None, sep=',', names=['Id', 'Content', 'Category']
    )

    pd.set_option('display.max_colwidth', None)
    # .pipe(remove_punctuation) \
    df['Content'] = df['Content'] \
        .pipe(remove_digits) \
        .pipe(remove_diacritics) \
        .pipe(remove_html_tags) \
        .pipe(remove_urls) \
        .pipe(remove_brackets) \
        .pipe(remove_brackets) \
        .pipe(remove_brackets) \
        .pipe(remove_whitespace) \
        .pipe(remove_regex)

    for idx, content in enumerate(df['Content']):
        if not content:
            df.drop(index=idx)

    df.to_csv("doc_class.csv", index=False, columns=['Content', 'Category'], sep=',')
    print(f"DONE: {df.values[0]}")

    # from ClassFile import ClassFile
    # content_list = sentence_splitter(content_list)
    # print("=====================================")
    # for st in content_list:
    #     print(f"* {st}", end="\n")
    #     print()
    #     if len(st) >= 90:
    #         ClassFile.to_txtfile(f"{st}\n", "process_dataset.txt", "a")


if __name__ == '__main__':
    main()
