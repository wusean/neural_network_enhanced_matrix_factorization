'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import argparse
from time import time

import numpy as np
from keras.layers import Embedding, Input, Dense, Flatten, Multiply, Concatenate
from keras.models import Model
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2

import DEMF
import MLP
from Dataset import Dataset
from evaluate import evaluate_model


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NNCF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=64,
                        help='Embedding size of DMF model.')
    parser.add_argument('--layers', nargs='?', default='[1024,512,256,128,64]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for DMF embeddings.')
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--dmf_pretrain', nargs='?', default='Pretrain/ml-1m_DMF4_[64,32,16,8]_1606876933.h5',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='Pretrain/ml-1m_MLP_[1024,512,256,128,64]_1608293197.h5',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()

#
# def init_normal(shape, name=None):
#     return initializations.normal(shape, scale=0.01, name=name)


def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Embedding layer
    # Deprecate: init and W_regularizer
    '''
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                  init=init_normal, W_regularizer=l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                  init=init_normal, W_regularizer=l2(reg_mf), input_length=1)

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] / 2, name="mlp_embedding_user",
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] / 2, name='mlp_embedding_item',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    '''

    DMF1_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim // 2, name='dmf1_embedding_user',
                                    embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_mf),
                                    input_length=1)
    DMF1_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim // 2, name='dmf1_embedding_item',
                                    embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_mf),
                                    input_length=1)
    DMF2_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim // 2, name='dmf2_embedding_user',
                                    embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_mf),
                                    input_length=1)
    DMF2_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim // 2, name='dmf2_embedding_item',
                                    embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_mf),
                                    input_length=1)

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] // 2, name="mlp_embedding_user",
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] // 2, name='mlp_embedding_item',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)

    # DMF part
    dmf1_user_latent = Flatten()(DMF1_Embedding_User(user_input))
    dmf1_item_latent = Flatten()(DMF1_Embedding_Item(item_input))
    dmf2_user_latent = Flatten()(DMF2_Embedding_User(user_input))
    dmf2_item_latent = Flatten()(DMF2_Embedding_Item(item_input))

    # Deprecate: merge
    '''mf_vector = merge([mf_user_latent, mf_item_latent], mode='mul')  # element-wise multiply'''
    vector_r = Multiply()([dmf1_user_latent, dmf1_item_latent])
    vector_l = Concatenate()([dmf2_user_latent, dmf2_item_latent])
    layer = Dense(mf_dim // 2, kernel_regularizer=l2(reg_layers[0]), activation='linear', name='pre-layer')
    vector_l = layer(vector_l)
    dmf_vector = Concatenate()([vector_r, vector_l])
    for idx in range(1, num_layer-4):
        # Deprecate: W_W_regularizer
        '''layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)'''
        layer = Dense(layers[idx] // 16, kernel_regularizer=l2(reg_layers[idx]), activation='relu', name='dmf_layer%d' % idx)
        dmf_vector = layer(dmf_vector)



    # MLP part
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))

    # Deprecate: merge
    '''mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode='concat')'''
    mlp_vector = Concatenate(axis=1)([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate DMF and MLP parts
    # mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    # mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)

    # Deprecate: merge
    '''predict_vector = merge([mf_vector, mlp_vector], mode='concat')'''
    predict_vector = Concatenate()([dmf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)

    return model


def load_pretrain_model(model, dmf_model, mlp_model, num_layers):
    # DMF embeddings
    dmf1_user_embeddings = dmf_model.get_layer('dmf1_user_embedding').get_weights()
    dmf1_item_embeddings = dmf_model.get_layer('dmf1_item_embedding').get_weights()
    dmf2_user_embeddings = dmf_model.get_layer('dmf2_user_embedding').get_weights()
    dmf2_item_embeddings = dmf_model.get_layer('dmf2_item_embedding').get_weights()
    model.get_layer('dmf1_embedding_user').set_weights(dmf1_user_embeddings)
    model.get_layer('dmf1_embedding_item').set_weights(dmf1_item_embeddings)
    model.get_layer('dmf2_embedding_user').set_weights(dmf2_user_embeddings)
    model.get_layer('dmf2_embedding_item').set_weights(dmf2_item_embeddings)

    # DMF layers
    dmf_layer_weights = dmf_model.get_layer('pre-layer').get_weights()
    model.get_layer('pre-layer').set_weights(dmf_layer_weights)
    for i in range(1, num_layers-4):
        dmf_layer_weights = dmf_model.get_layer('layer%d' % i).get_weights()
        model.get_layer('dmf_layer%d' % i).set_weights(dmf_layer_weights)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' % i).get_weights()
        model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)

    # Prediction weights
    dmf_prediction = dmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((dmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = dmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5 * new_weights, 0.5 * new_b])
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
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    dmf_pretrain = args.dmf_pretrain
    mlp_pretrain = args.mlp_pretrain

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("NNCF arguments: %s " % (args))
    model_out_file = 'Pretrain/%s_NNCF_%d_%s_%d.h5' % (args.dataset, mf_dim, args.layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

    # Load pretrain model
    if dmf_pretrain != '' and mlp_pretrain != '':
        dmf_model = DEMF.get_model(num_users, num_items, layers=[64, 32, 16, 8], reg_layers=[0, 0, 0, 0])
        dmf_model.load_weights(dmf_pretrain)
        mlp_model = MLP.get_model(num_users, num_items, layers, reg_layers)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, dmf_model, mlp_model, len(layers))
        print("Load pretrained DMF (%s) and MLP (%s) models done. " % (dmf_pretrain, mlp_pretrain))

    # Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out > 0:
        model.save_weights(model_out_file, overwrite=True)

    # Training model
    for epoch in range(num_epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)

        # Training
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
        print("The best NNCF model is saved to %s" % (model_out_file))
