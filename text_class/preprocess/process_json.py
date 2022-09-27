import json
import os

from common.ClassFile import ClassFile


def dump(_content, _path):
    _file_path = os.path.dirname(os.path.realpath(__file__)) + _path
    ClassFile.to_txtfile(
        data=_content + "\n",
        file_=_file_path,
        mode="a")


def main():
    dataset_list = list()
    with open("dataset/doc_classification.json", "r") as read_file:
        file_data_list = read_file.readlines()
        for data in file_data_list:
            json_data = json.loads(data)
            dataset_list.append(json_data)

    for data_item in dataset_list:
        label_list = ",".join(data_item['annotation']['labels'])
        content = f"\"{data_item['content']}\",{label_list}"
        dump(content, "/dataset/doc_classification.csv")


if __name__ == '__main__':
    main()
