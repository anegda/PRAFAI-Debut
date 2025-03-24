import argparse

from flair.datasets import ClassificationCorpus, CSVClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings, DocumentRNNEmbeddings, TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, CharacterEmbeddings, BertEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train a Flair model for AF detection in medical reports')
    parser.add_argument("--input", help="Path to the input folder with (train / dev / test).")
    parser.add_argument("--output", help="Path to output directory for the model.")

    args = vars(parser.parse_args())
    data_folder = args['input']
    output_dir = args['output']

    #CREAMOS EL CLASSIFCATION CORPUS:  https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_6_CORPUS.md#reading-a-text-classification-dataset
    column_name_map = {4: "text", 5: "label_topic"}
    corpus = CSVClassificationCorpus(data_folder, column_name_map, skip_header=True, delimiter=',', label_type="class")
    label_dict = corpus.make_label_dictionary(label_type="class")

    #OBTENEMOS LOS EMBEDDINGS PREENTRENADOS
    word_embeddings = [
        # VECTORES:
        # FASTTEXT ENTRENADOS EN WIKIPEDIA
        WordEmbeddings('es'),
        # WORDEMBEDDINGS PERSONALIZADOS
        #WordEmbeddings(dir_embeddings),

        # CHARACTER EMBEDDIGNS
        #CharacterEmbeddings(),

        # FLAIR EMBEDDINGS (se necesitan tanto fordward como backward):
        # PERSONALIZADAS
        #FlairEmbeddings(forward_dir),
        #FlairEmbeddings(backward_dir),

        # GENERADOS CON WIKIPEDIA
        FlairEmbeddings('es-forward'),
        FlairEmbeddings('es-backward')
    ]
    # 4. initialize transformer document embeddings (many models are available)
    document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(word_embeddings,
                                                                       hidden_size=256,
                                                                       reproject_words=True,
                                                                       rnn_type='LSTM',
                                                                       )
    #CREAMOS EL TEXT CLASSIFIER
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type="class")

    #INICIALIZAOS EL TEXT CLASSIFIER TRAINER
    trainer = ModelTrainer(classifier, corpus)

    #EMPEZAMOS EL ENTRENAMIENTO
    trainer.train(output_dir, patience=10, mini_batch_size=50, learning_rate=0.01, max_epochs=100, train_with_dev=False)

