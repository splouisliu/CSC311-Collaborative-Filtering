from utils import *
import matplotlib.pyplot as plt
import numpy as np

# Function To Calculate Probability c_ij = 1 given theta, beta
def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    # Go through the entire raw dataset
    for i in range(len(data['is_correct'])):
        c_ij = data['is_correct'][i]
        beta_j = beta[data['question_id'][i]]
        theta_i = theta[data['user_id'][i]]
        log_lklihood+= c_ij*(theta_i-beta_j)-np.log(1+np.exp(theta_i-beta_j))    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood

def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Convert each column in dataset to numpy arrays for easy operations
    question_id = np.array(data['question_id'])
    user_id = np.array(data['user_id'])
    c_ij = np.array(data['is_correct'])

    # Create new theta and betas vectors for final results
    theta_new = np.zeros_like(theta)
    beta_new = np.zeros_like(beta)
    
    for user in range(len(theta)):
        # Go through beta vector and only put beta[j] in beta_important if is_correct[user][j] is NOT NULL (ignore null data)
        # In other words, extract all beta[question_id] for all question_id where user_id == user (current iteration user)
        beta_important = np.take(beta,question_id[np.where(user_id==user)])
        # Perform update (gradient ascent of log likelihood)
        theta_new[user] = theta[user] + lr*(np.sum(c_ij[np.where(user_id==user)]) - np.sum(sigmoid(theta[user]-beta_important)))
    
    for question in range(len(beta)):
        # Go through theta vector and only put theta[i] in theta_important if is_correct[i][question] is NOT NULL (ignoring null data)
        # In other words, extract all theta[user] for all users where question_id == question (current iteration question)
        theta_important = np.take(theta,user_id[np.where(question_id==question)])
        # Perform update (gradient ascent of log likelihood)
        beta_new[question] = beta[question] + lr*(-np.sum(c_ij[np.where(question_id==question)]) + np.sum(sigmoid(theta_important-beta[question])))
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta_new, beta_new


def irt(train_data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    np.random.seed(818)
    # theta = np.random.rand(max(np.array(train_data['user_id']))+1)
    # beta = np.random.rand(max(np.array(train_data['question_id']))+1)
    theta = np.zeros(max(np.array(train_data['user_id']))+1)
    beta = np.zeros(max(np.array(train_data['question_id']))+1)
    
    train_acc_1st = []
    train_llds = []
    val_acc_lst = []
    val_llds = []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(train_data, theta=theta, beta=beta)
        train_llds.append(train_neg_lld)
        score = evaluate(data=train_data, theta=theta, beta=beta)
        train_acc_1st.append(score)
        # print("Train NLLK: {} \t Train Score: {}".format(train_neg_lld, score))
        
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_llds.append(val_neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        # print("Val NLLK: {} \t Val Score: {}".format(val_neg_lld, score))
        
        theta, beta = update_theta_beta(train_data, lr, theta, beta)
        # print("Theta: {} \t Beta: {}".format(theta, beta))

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_acc_1st ,val_acc_lst, train_llds, val_llds

def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
            
    # Setting a seed        
    np.random.seed(818)
    
    # # Grid Search
    # lrs = {0.01}
    # iterations ={5,7,10,12,15,17,20,22,25,30}
    # for lr in lrs:
    #     for iteration in iterations:
    #         theta, beta, train_accs, val_accs, train_llds, val_llds = irt(train_data, val_data, lr, iteration)
    #         print("Final Validation Accuracy (LR={}, IT={}): {}".format(lr,iteration,val_accs[-1]))
    #         #print("Final Train Accuracy (LR={}, IT={}): {}".format(lr,iteration,train_accs[-1]))

    # Defining the Hyperparameters
    lr = 0.01
    num_iterations = 15
    
    # Training the Model
    theta, beta, train_accs, val_accs, train_llds, val_llds = irt(train_data, val_data, lr, num_iterations)
    
    # # Obtaining the Results
    # iteration_list = list(range(1,num_iterations+1))
    # final_val_acc = evaluate(val_data, theta, beta)
    # final_test_acc = evaluate(test_data, theta, beta)
    # print(lr)
    # print(num_iterations)
    # print("The Final Validation Accuracy is: {}".format(final_val_acc))
    # print("The Final Test Accuracy is: {}".format(final_test_acc))
    # plt.title("Negative Log Likelihood vs Iterations")
    # plt.xlabel("Iterations")
    # plt.ylabel("Negative Log Likelihood")
    # plt.xticks(iteration_list)
    # plt.plot(iteration_list,train_llds,label = "Train")
    # plt.plot(iteration_list,val_llds,label = 'Validation')
    # plt.legend()
    # plt.show()
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)   
    questions = [33,66,99]
    
    prob_correct_1 = []
    for user in range(len(theta)):
        prob_correct_1.append(sigmoid(theta[user]-beta[questions[0]]))
    
    prob_correct_2 = []
    for user in range(len(theta)):
        prob_correct_2.append(sigmoid(theta[user]-beta[questions[1]]))
        
    prob_correct_3 = []
    for user in range(len(theta)):
        prob_correct_3.append(sigmoid(theta[user]-beta[questions[2]]))
    
    plt.title("Probability of Correct vs Theta For Specific Questions")
    plt.scatter(theta, prob_correct_1, s=1, label='Question 33')
    plt.scatter(theta, prob_correct_2, s=1, label='Question 66')
    plt.scatter(theta, prob_correct_3, s=1, label='Question 99')
    plt.xlabel("Theta")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()
    
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

if __name__ == "__main__":
    main()