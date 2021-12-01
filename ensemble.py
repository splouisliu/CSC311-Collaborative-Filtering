from utils import *
from sklearn.impute import KNNImputer

from item_response import irt, sigmoid
from neural_network import device, AutoEncoder, train as train_autoencoder

import torch
from torch.autograd import Variable
import numpy as np
import random as r

BASE_PATH = "./data"


## Data bootstrapping
##
def bootstrap(data, sets=3):
    '''takes data dict from load_train_csv and 
    creates randomized training sets'''
    r.seed(2021)
    
    boots = {}
    
    for i in range(sets):
        boots[f"resample{i}"]={
        "user_id": [],
        "question_id": [],
        "is_correct": []
        }

        for j in range(len(data["is_correct"])):
            random_idx= r.randint(0, len(data["is_correct"])-1)
            boots[f"resample{i}"]['user_id'].append(data['user_id'][random_idx])
            boots[f"resample{i}"]['question_id'].append(data['question_id'][random_idx])
            boots[f"resample{i}"]['is_correct'].append(data['is_correct'][random_idx])
    return boots


def dict_to_matrix(data):
    matrix = np.full([542, 1774], np.nan)
    
    for i in range(len(data["is_correct"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        c = data["is_correct"][i]
        
        matrix[user_id][question_id] = c
    
    return matrix


## Ensemble
##
class Ensemble():
    def __init__(self):
        self.knn_matrix = None
        self.irt_alpha = None
        self.irt_beta = None
        self.nn_model = None
        self.nn_zero_train_matrix = None
        

def train_knn(model, train_matrix, k):
    # Impute matrix by user
    nbrs = KNNImputer(n_neighbors=k)
    model.knn_matrix = nbrs.fit_transform(train_matrix)


def train_irt(model, train_data, valid_data, lr, num_iterations):
    model.theta, model.beta, _, _, _, _ = irt(train_data, valid_data, lr, num_iterations)


def train_nn(model, train_matrix, valid_data, k, lr, num_epoch, lamb):
    # Prep training data
    model.nn_zero_train_matrix = train_matrix.copy()
    model.nn_zero_train_matrix[np.isnan(train_matrix)] = 0
    model.nn_zero_train_matrix = torch.FloatTensor(model.nn_zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)
    
    # Train AutoEncoder model
    model.nn_model = AutoEncoder(train_matrix.shape[1], k)
    model.nn_model.to(device)
    train_autoencoder(model.nn_model, lr, lamb, train_matrix, model.nn_zero_train_matrix, valid_data, num_epoch)


def train(model, knn_k, irt_lr, irt_iterations, nn_k, nn_lr, nn_epochs, nn_lamb):
    # Load and bootstrap training data
    train_boot = bootstrap(load_train_csv(BASE_PATH))

    knn_train = train_boot["resample0"]
    knn_train_matrix = dict_to_matrix(knn_train)
    irt_train = train_boot["resample1"]
    nn_train = train_boot["resample2"]
    nn_train_matrix = dict_to_matrix(nn_train)
    
    # Load validation data
    valid_data = load_valid_csv(BASE_PATH)
    
    # Train
    train_knn(model, knn_train_matrix, knn_k)
    train_irt(model, irt_train, valid_data, irt_lr, irt_iterations)
    train_nn(model, nn_train_matrix, valid_data, nn_k, nn_lr, nn_epochs, nn_lamb)


def predict(model, query='test', weights=[1/3, 1/3, 1/3], threshold=0.5):
    # Load query data
    if query=='validation':
        query_data = load_valid_csv(BASE_PATH)
    elif query == 'test':
        query_data = load_public_test_csv(BASE_PATH)
    
    # Predict
    knn_preds = []
    irt_preds = []
    nn_preds = []
    
    for i in range(len(query_data["is_correct"])):
        user_id = query_data["user_id"][i]
        question_id = query_data["question_id"][i]
        
        # Predict on KNN
        knn_preds.append(model.knn_matrix[user_id][question_id])
        
        # Predict on IRT
        irt_preds.append(sigmoid(model.theta[user_id]-model.beta[question_id]))
        
        # Predict on NN
        inputs = Variable(model.nn_zero_train_matrix[user_id]).unsqueeze(0).to(device)
        output = model.nn_model(inputs)
        guess = output[0][question_id].item()
        nn_preds.append(guess)
    
    # Bag predictions
    bagged_preds = weights[0]*np.array(knn_preds) + weights[1]*np.array(irt_preds) + weights[2]*np.array(nn_preds) >= threshold
    
    return bagged_preds
    
        
def evaluate(predictions, query="test"):
    # Load query data
    if query == "validation":
        query_data = load_valid_csv(BASE_PATH)
    elif query == "test":
        query_data = load_public_test_csv(BASE_PATH)
    
    # Calculate accuracy
    acc = sum(predictions == query_data["is_correct"])/len(predictions)
    
    return acc


def main():
    # Set hyperparameters
    knn_k = 11

    irt_lr = 0.01
    irt_iterations = 40

    nn_k = 50
    nn_lr = 0.01
    nn_epochs = 50
    nn_lamb = 0.001

    # Train model
    model = Ensemble()
    train(model, knn_k, irt_lr, irt_iterations, nn_k, nn_lr, nn_epochs, nn_lamb)
    
    # Evaluate on validation set
    pred_valid = predict(model, query="validation")
    acc_valid = evaluate(pred_valid, query="validation")

    # Evaluate on test set
    pred_test= predict(model, query="test")
    acc_test = evaluate(pred_test, query="test")

    # Results
    print("Validation accuracy=", acc_valid)
    print("Test accuracy=", acc_test)
    

if __name__ == "__main__":
    main()
