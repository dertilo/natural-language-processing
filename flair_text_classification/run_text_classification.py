from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


corpus: Corpus = TREC_6()

label_dict = corpus.make_label_dictionary()

word_embeddings = [
    WordEmbeddings("glove"),
]

document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
    word_embeddings,
    hidden_size=512,
    reproject_words=True,
    reproject_words_dimension=256,
)

classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

trainer = ModelTrainer(classifier, corpus)

trainer.train(
    "resources/text_clf",
    learning_rate=0.1,
    mini_batch_size=32,
    anneal_factor=0.5,
    patience=5,
    max_epochs=50,
)
