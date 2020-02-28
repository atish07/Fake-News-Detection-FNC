import sys
import numpy as np
import pandas as pd
from download import downloadfile
from termfeat import terms
import nltk
from tf_idf import getTFID
from keras.utils import np_utils
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features, clean, get_tokenized_lemmas
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from similarity import find_similarity
from utils.system import parse_params, check_version
import os
from keras.models import Sequential, Model
from keras.layers import concatenate, Dense, Dropout, LSTM, GRU, Flatten, Conv1D, AveragePooling1D, Input
np.random.seed(seed=0)
downloadfile()
def generate_features(stances, dataset, name):
    h, b, y = [], [], []

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
    X1 = getTFID(name, "features/tfidf."+name+".npy")
    X_similarity = find_similarity(h, b, "features/bertsim."+name+".npy", name)
    X_overlap = gen_or_load_feats(
        word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(
        refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(
        polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(
        hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_overlap, X_refuting, X_polarity, X_hand, X_similarity]
    X_tfidf = X1
    return X, X_tfidf, y


if __name__ == "__main__":
    check_version()
    parse_params()
    train_comb = pd.read_csv(
        "targetdir/bertembed/traincombined_bert.csv").drop('Unnamed: 0', axis=1)
    comp_comb = pd.read_csv(
        "targetdir/bertembed/compcombined_bert.csv").drop('Unnamed: 0', axis=1)
    train_head = pd.read_csv(
        "targetdir/bertembed/trainheadline_bert.csv").drop('Unnamed: 0', axis=1)
    train_body = pd.read_csv(
        "targetdir/bertembed/trainbody_bert.csv").drop('Unnamed: 0', axis=1)
    comp_head = pd.read_csv(
        "targetdir/bertembed/compheadline_bert.csv").drop('Unnamed: 0', axis=1)
    comp_body = pd.read_csv(
        "targetdir/bertembed/compbody_bert.csv").drop('Unnamed: 0', axis=1)
    #Load the training dataset
    d = DataSet()

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")

    X_train, Xtrain_TFIDF, y_train = generate_features(d.stances, d, "train")
    X_competition, Xcomp_TFIDF, y_competition = generate_features(
        competition_dataset.stances, competition_dataset, "comp")
    y_train = np_utils.to_categorical(y_train)

    Xtrain_wordfeats = np.c_[Xtrain_TFIDF, train_comb, train_body,train_head]
    Xcomp_wordfeats = np.c_[Xcomp_TFIDF, comp_comb, comp_body,comp_head]

    # X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    # X_competition = np.reshape(X_competition, (X_competition.shape[0], 1,X_competition.shape[1]))

    # Xtrain_wordfeats = np.reshape(Xtrain_wordfeats, (Xtrain_wordfeats.shape[0], 1, Xtrain_wordfeats.shape[1]))
    # Xcomp_wordfeats= np.reshape(Xcomp_wordfeats, (Xcomp_wordfeats.shape[0], 1,Xcomp_wordfeats.shape[1]))

    feats = Input(shape=(X_train.shape[1],))
    f_layer1 = Dense(500, activation='tanh')(feats)
    f_droplayer1 = Dropout(0.8)(f_layer1)
    f_layer2 = Dense(300, activation='sigmoid')(f_droplayer1)
    f_droplayer2 = Dropout(0.8)(f_layer2)
    f_layer3 = Dense(100, activation='sigmoid')(f_droplayer2)
    f_out = Dense(4, activation='softmax', name='sec')(f_layer3)

    word_feats = Input(shape=(Xtrain_wordfeats.shape[1],))
    wf_layer1 = Dense(500, activation='tanh')(word_feats)
    wf_droplayer1 = Dropout(0.8)(wf_layer1)
    wf_layer2 = Dense(500, activation='sigmoid')(wf_droplayer1)
    wf_droplayer2 = Dropout(0.1)(wf_layer2)
    wf_layer3 = Dense(100, activation='sigmoid')(wf_droplayer2)


    combine = concatenate([f_layer3, wf_layer3])
    out1 = Dense(100, activation='tanh')(combine)
    out2 = Dense(300, activation='sigmoid')(out1)
    out3 = Dense(100, activation='sigmoid')(out2)
    final_out = Dense(4, activation='softmax', name="Output_Layer")(out3)

    model = Model(inputs=[feats, word_feats], outputs=[final_out, f_out])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  loss_weights=[0.5, 0.5], metrics=['accuracy'])
    model.fit([X_train, Xtrain_wordfeats], [y_train, y_train],
              epochs=15, batch_size=128, validation_split=0.2)


    y_preds = model.predict([X_competition, Xcomp_wordfeats])

    preds = []
    for i in range(len(y_preds[0])):
        preds.append((y_preds[0][i]+y_preds[1][i]))
    predicted = [LABELS[int(np.argmax(a))] for a in preds]
    actual = [LABELS[int(a)] for a in y_competition]
    report_score(actual, predicted)

