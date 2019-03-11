import argparse
import logging
from utils.logger import logger_initialization
import pandas as pd
import sys
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import itertools
from keras.models import Sequential
from keras import layers
from datetime import datetime
import pickle
from keras import backend
from sklearn.metrics import confusion_matrix
from keras.regularizers import l1_l2
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer


# making sure we have the right path for the data based on the system where it is run
# local computer or Qkliview server
if sys.platform == "darwin" or sys.platform == "win32":
    if sys.platform == "win32":
        path = 'D:\dataset\gsq'
    else:
        path = '/Volumes/dataset/gsq'
# Respublica
else:
    path = 'dataset/'


model_name = {'nn': 'neural_network', 'lr': 'logistic_regression'}


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True, ext=''):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    ext:          training or testing extension

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    time_name = datetime.now().strftime('%Y%m%d%H%M%S')
    img_path = os.path.join(path, 'results')
    img_file_name = model_name[title] + '_' + ext + '_' + time_name + '.png'
    img_path = os.path.join(img_path, img_file_name)
    plt.savefig(img_path)
    logging.getLogger('regular').info("confusion matrix img: {0}".format(img_path))

    # only show if local computer or windows server. Not show in Respublica
    if sys.platform == "darwin" or sys.platform == "win32":
        plt.show()


def train_model(classifier, train_data, train_label, test_data, test_label, model=''):

    if model == 'lr':
        # fit the training dataset on the classifier
        classifier.fit(train_data, train_label)

        logging.getLogger('regular.time').debug('finished training model')

        # predict the labels on validation dataset
        train_predictions = classifier.predict(train_data)
        test_predictions = classifier.predict(test_data)

        time_name = datetime.now().strftime('%Y%m%d%H%M%S')
        file_name = model_name[model] + '_' + time_name + '.json'
        model_output_dir = 'models/' + file_name
        model_output_dir = os.path.join(path, model_output_dir)
        pickle.dump(classifier, open(model_output_dir, 'wb'))

        training_score = accuracy_score(train_predictions, train_label)
        testing_score = accuracy_score(test_predictions, test_label)

    elif model == 'nn':

        # fit the training dataset on the classifier
        history = classifier.fit(train_data, train_label, epochs=150, verbose=True)

        logging.getLogger('regular.time').debug('finished training model')

        # predict the labels on validation dataset
        train_predictions = classifier.predict(train_data)
        test_predictions = classifier.predict(test_data)

        train_predictions = train_predictions.argmax(axis=-1)
        test_predictions = test_predictions.argmax(axis=-1)

        # serialize model to JSON
        time_name = datetime.now().strftime('%Y%m%d%H%M%S')
        file_name = model_name[model] + '_' + time_name + '.json'
        model_output_dir = 'models/' + file_name
        model_output_dir = os.path.join(path, model_output_dir)
        weight_file_name = model_name[model] + '_weights_' + time_name + '.h5'
        weight_model_output_dir = 'models/' + weight_file_name
        weight_model_output_dir = os.path.join(path, weight_model_output_dir)

        history_file_name = model_name[model] + '_history_' + time_name + '.csv'
        history_model_output_dir = 'models/' + history_file_name
        history_model_output_dir = os.path.join(path, history_model_output_dir)
        history = pd.DataFrame.from_dict(history.history)
        history.to_csv(history_model_output_dir, index=False)

        msg = '-' * 10
        logging.getLogger('regular').info(msg=msg)
        logging.getLogger('regular').info(msg=msg)

        model_json = classifier.to_json()
        with open(model_output_dir, 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        classifier.save_weights(weight_model_output_dir)

        msg = '-' * 10
        logging.getLogger('regular').info(msg=msg)
        logging.getLogger('regular').info(msg=msg)

        training_score = classifier.evaluate(train_data, train_label, verbose=0)[1]
        testing_score = classifier.evaluate(test_data, test_label, verbose=0)[1]

        logging.getLogger('regular.time').debug(msg='model summary')
        logging.getLogger('regular.time').debug(msg=classifier.summary())

        backend.clear_session()

    msg = 'training accuracy: {0:.2f}'.format(training_score * 100)
    logging.getLogger('regular').info(msg)
    msg = 'testing accuracy: {0:.2f}'.format(testing_score * 100)
    logging.getLogger('regular').info(msg)

    msg = 'Saved model to disk time = {0}'.format(time_name)
    logging.getLogger('regular.time').info(msg=msg)

    return train_predictions, test_predictions


def run_model(x_data, y_data, model_flag, preproc_flag, embed_layer=False):

    # embbed layer off
    el = 'elof'

    # create the training and testing dataset
    query_train, query_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=1000)

    y_train = y_train.values.astype(int)
    y_test = y_test.values.astype(int)

    # count Vectorizer
    if preproc_flag == 'cv':
        # simple BOW model
        vectorizer = CountVectorizer()
        vectorizer.fit(query_train)
        # vectorize the queries
        x_train = vectorizer.transform(query_train)
        x_test = vectorizer.transform(query_test)

    # word embedding
    elif preproc_flag == 'we':
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(query_train)

        x_train = tokenizer.texts_to_sequences(query_train)
        x_test = tokenizer.texts_to_sequences(query_test)

        maxlen = 300

        x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
        x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

    elif preproc_flag == 'tfidf':
        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        tfidf_vect.fit(query_train)
        x_train = tfidf_vect.transform(query_train)
        x_test = tfidf_vect.transform(query_test)

    if model_flag == 'lr':
        classifier = LogisticRegression(n_jobs=-1)

    elif model_flag == 'nn':

        if embed_layer:

            # embbed layer on
            el = 'elon'

            embedding_dim = 50
            vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

            classifier = Sequential()
            classifier.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
            # model.add(layers.Flatten())
            classifier.add(layers.GlobalMaxPool1D())
            classifier.add(layers.Dense(10, activation='relu'))
            classifier.add(layers.Dense(1, activation='sigmoid'))

        else:

            # embbed layer off
            el = 'elof'

            input_dim = x_train.shape[1]  # Number of features

            classifier = Sequential()
            classifier.add(layers.Dense(100, input_dim=input_dim, activation='relu'))
            classifier.add(Dropout(0.2))
            classifier.add(layers.Dense(80, input_dim=input_dim, activation='relu', kernel_regularizer=l1_l2(0.01)))
            classifier.add(Dropout(0.2))
            classifier.add(layers.Dense(50, input_dim=input_dim, activation='relu', kernel_regularizer=l1_l2(0.01)))
            classifier.add(Dropout(0.2))
            classifier.add(layers.Dense(20, activation='sigmoid'))

            x_test = x_test.todense()
            x_train = x_train.todense()

        classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # obtain model results
    train_pred, test_pred = train_model(classifier=classifier, train_data=x_train, train_label=y_train,
                                        test_data=x_test, test_label=y_test, model=model_flag)

    ext = preproc_flag + '_' + el + 'train'
    plot_confusion_matrix(confusion_matrix(y_train, train_pred), target_names=np.unique(y_train), title=model_flag,
                          ext=ext)
    ext = preproc_flag + '_' + el + 'test'
    plot_confusion_matrix(confusion_matrix(y_test, test_pred), target_names=np.unique(y_test), title=model_flag,
                          ext=ext)


def main():
    # get the the path for the input file argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', help='dataset')
    parser.add_argument('-m', '--model', help='model to run i.e. neural networks, support vector machine or '
                                              'logistic regression', choices=['nn', 'svm', 'lr'], default='nn')
    parser.add_argument('-p', '--preproc', help='preprocessing flag use to use countVectorizer or WordEmbedding',
                        choices=['cv', 'we', 'tfidf'], default='we')
    parser.add_argument('-e', '--embed_layer', action='store_true', help='embedding layer for keras')
    parser.add_argument('-l', '--log', dest='logLevel', choices=['DEBUG', 'INFO', 'ERROR'], type=str.upper,
                        help='Set the logging level')
    args = parser.parse_args()

    logger_initialization(log_level=args.logLevel)
    logging.getLogger('line.regular.time').info('Running GSQ script')

    # print the arguments of the scripts
    logging.getLogger('regular').debug(args)

    # creating directory path for the file
    queries_path = os.path.join(path, args.data_file)
    # reading file
    msg = 'reading file = {0}'.format(queries_path)
    logging.getLogger('regular.time').info(msg)
    data = pd.read_csv(queries_path)

    # removing irrelevant columns
    data.drop(['id', 'timestamp_usec', 'timestamp', 'FOLDER'], axis=1, inplace=True)

    # renaming columns
    data.columns = list(['query', 'label'])
    data.head()

    # check if cross validation flag is set
    run_model(x_data=data['query'], y_data=data['label'], model_flag=args.model, preproc_flag=args.preproc,
              embed_layer=args.embed_layer)


if __name__ == '__main__':
    main()
