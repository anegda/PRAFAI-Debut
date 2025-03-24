import argparse

from flair.models import TextClassifier
from flair.datasets import DataLoader, CSVClassificationCorpus
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to evaluate the Flair model')
    parser.add_argument("--input", help="Path to the input folder with (train / dev / test).")
    parser.add_argument("--model_input", help="Path to the model .pt file.")

    args = vars(parser.parse_args())
    data_folder = args['input']
    model_pt = args['model_input']

    model = TextClassifier.load(model_pt)

    column_name_map = {4: "text", 5: "label_topic"}
    corpus = CSVClassificationCorpus(data_folder, column_name_map, skip_header=True, delimiter=',', label_type="class")

    result = model.evaluate(corpus.test, mini_batch_size=50, out_path=f"predictionsInter1.txt", gold_label_type="class")
    print(result)