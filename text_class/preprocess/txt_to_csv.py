import os

from common.ClassFile import ClassFile
from common.ExtractorService import ExtractorService
from common.SpacyModel import SpacyModel

MAX_LENGTH = 500
DATA_PATH = "./../dataset/"
DATA_OUTPUT = ".csv"


def clean_file(content):
    content = ExtractorService.clean_doc(content)
    content = ExtractorService.simplify(content)
    return content


def dump(content, category):
    ClassFile.to_txtfile(
        data=f"{content},{category}\n",
        file_=DATA_OUTPUT,
        mode="a+",
        encoding="utf-8")


def load_data():
    file_list = ClassFile.list_files_like(DATA_PATH, "txt")
    file_filter = ClassFile.filter_by_size(file_list)
    print(f"TOTAL FILE: {len(file_filter)}")

    return file_filter


def clean_words(doc: str):
    name_list = set()
    doc_nlp = SpacyModel.getInstance().getModel('es')(doc)
    for token in doc_nlp:
        if (
                token.pos_ == r'PROPN' or
                token.pos_ == r'NOUN' or
                token.pos_ == r'ADJ' or
                token.pos_ == r'ADV' or
                token.pos_ == r'AUX' or
                token.pos_ == r'INTJ' or
                token.pos_ == r'VERB'
        ):
            name_list.add(token.text)

    return list(name_list)


def main():
    print("loading data...")
    dataset = load_data()
    left = len(dataset)
    for file in dataset:
        class_id = ''
        data_decoded = ClassFile.get_text(file)
        data_decoded = clean_file(data_decoded)
        try:
            class_path, _ = os.path.split(file)
            _, class_id = os.path.split(class_path)
            word_list = clean_words(data_decoded)
            if word_list:
                data = ' '.join(word_list)[:300]
                if data:
                    dump(f"\"{data}\"", category=class_id)
                else:
                    print(f"DATA NOT FOUND OR MISSING: {file}")
            else:
                print(f"WORDS NOT FOUND OR MISSING: {file}")
        except Exception:
            print(f"EXCEPTION: {file}")

        left -= 1
        print(f"LEFT: {left} / {class_id}")


if __name__ == '__main__':
    main()
