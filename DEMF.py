'''
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''

import numpy as np
# import theano
# import theano.tensor as T
import keras
from keras import backend as K
# from keras import initializations
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, Dropout, Concatenate, Multiply, Add, Activation
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run DMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='pinterest-20',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[8,4,2,1]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=5,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


'''
def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)
'''


def get_model(num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Deprecate: init and W_regularizer
    '''
    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]/2, name = 'user_embedding',
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]/2, name = 'item_embedding',
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)
    '''

    DMF1_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] // 2, name='dmf1_user_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)
    DMF1_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] // 2, name='dmf1_item_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)

    DMF2_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] // 2, name='dmf2_user_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)
    DMF2_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] // 2, name='dmf2_item_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)

    # Crucial to flatten an embedding vector!
    dmf1_user_latent = Flatten()(DMF1_Embedding_User(user_input))
    dmf1_item_latent = Flatten()(DMF1_Embedding_Item(item_input))

    dmf2_user_latent = Flatten()(DMF2_Embedding_User(user_input))
    dmf2_item_latent = Flatten()(DMF2_Embedding_Item(item_input))

    # The 0-th layer is the concatenation of embedding layers 1#
    # vector = Multiply()([user_latent, item_latent])

    # The 0-th layer is the concatenation of embedding layers 2#
    # vector_r = Multiply()([user_latent, item_latent])
    # vector_l = Concatenate()([user_latent, item_latent])
    # layer = Dense(layers[1], kernel_regularizer=l2(reg_layers[0]), activation='linear', name='pre-layer')
    # vector_l = layer(vector_l)
    # vector = Add()([vector_r, vector_l])
    # vector = Activation(activation='relu')(vector)


    # The 0-th layer is the concatenation of embedding layers 3#
    vector_r = Multiply()([dmf1_user_latent, dmf1_item_latent])
    vector_l = Concatenate()([dmf2_user_latent, dmf2_item_latent])

    # linear learning layer 3.1#
    # for idx in range(0, 2):
    #     layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[0]), activation='linear', name='pre-layer%d' % idx)
    #     vector_l = layer(vector_l)

    layer = Dense(layers[0] // 2, kernel_regularizer=l2(reg_layers[0]), activation='linear', name='pre-layer')
    vector_l = layer(vector_l)
    vector = Concatenate()([vector_r, vector_l])


    # The 0-th layer is the concatenation of embedding layers 4#
    # vector_r = Multiply()([user_latent, item_latent])
    # vector_l = Concatenate()([user_latent, item_latent])
    # vector = Concatenate()([vector_r, vector_l])


    # MLP layers (Dual-wise)
    for idx in range(1, num_layer-3):
        # Deprecate: W_W_regularizer
        '''layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)'''
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
        vector = layer(vector)

    # Final prediction layer

    # Deprecate: init
    '''prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(vector)'''
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vector)

    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)

    return model


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            '''while train.has_key((u, j))'''
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("DMF arguments: %s " % (args))
    model_out_file = 'Pretrain/%s_DMF4_%s_%d.h5' % (args.dataset, args.layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)

        # Training
        # UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
        '''
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        '''
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best DMF model is saved to %s" % (model_out_file))
