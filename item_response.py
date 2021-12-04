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
    # Initialize Log-Likelihood
    log_lklihood = 0.

    # Iterate through entire dictionary dataset and get beta_j, theta_i and c_ij
    # Iterating through dictionary directly means not worrying about c_ij=nan case
    for iter in range(len(data['is_correct'])):
        c_ij = data['is_correct'][iter]
        i = data['user_id'][iter]
        j = data['question_id'][iter]
        beta_j = beta[j]
        theta_i = theta[i]
        
        # Update Log-Likelihood
        log_lklihood = log_lklihood + c_ij*(theta_i)-c_ij*(beta_j)-np.log(1+np.exp(theta_i-beta_j))    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood

def update_theta_beta(data, lr, theta, beta, iterations):
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
    :iterations: int
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

    # Perform Alternating Gradient Descent
    if (iterations%2):  # Update Theta
        print("Iteration: " + str(iterations) + " (Updating Theta)")
        
        # Create a container to store new theta vector and populate it
        theta_new = np.zeros_like(theta)
        for user in range(len(theta)):
            # Go through beta vector and only put beta[j] in beta_important if is_correct[user][j] is NOT NULL (ignore null data)
            # In other words, extract all beta[question_id] for all question_ids where user_id == user (current iteration user)
            beta_important = np.take(beta,question_id[np.where(user_id==user)])
            # Perform update (gradient ascent of log likelihood)
            theta_new[user] = theta[user] + lr*(np.sum(c_ij[np.where(user_id==user)]) - np.sum(sigmoid(theta[user]-beta_important)))
        
        return theta_new, beta
    
    else:   # Update Beta
        print("Iteration: " + str(iterations) + " (Updating Beta)")
        
        # Create a container to store new beta vector and populate it
        beta_new = np.zeros_like(beta)
        for question in range(len(beta)):
            # Go through theta vector and only put theta[i] in theta_important if is_correct[i][question] is NOT NULL (ignoring null data)
            # In other words, extract all theta[user] for all users where question_id == question (current iteration question)
            theta_important = np.take(theta,user_id[np.where(question_id==question)])
            # Perform update (gradient ascent of log likelihood)
            beta_new[question] = beta[question] + lr*(-np.sum(c_ij[np.where(question_id==question)]) + np.sum(sigmoid(theta_important-beta[question])))
        
        return theta, beta_new
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


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
    # np.random.seed(818)
    # theta = np.random.rand(max(np.array(train_data['user_id']))+1)
    # beta = np.random.rand(max(np.array(train_data['question_id']))+1)
    theta = np.zeros(max(np.array(train_data['user_id']))+1)
    beta = np.zeros(max(np.array(train_data['question_id']))+1)
    
    train_acc_list = []
    train_llds = []
    val_acc_list = []
    val_llds = []

    for iter in range(iterations):
        train_neg_lld = neg_log_likelihood(train_data, theta=theta, beta=beta)
        train_llds.append(train_neg_lld)
        train_score = evaluate(data=train_data, theta=theta, beta=beta)
        train_acc_list.append(train_score)
        print("Train NLLK: {} \t Train Score: {}".format(train_neg_lld, train_score))
        
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_llds.append(val_neg_lld)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_list.append(val_score)
        print("Val NLLK: {} \t Val Score: {}".format(val_neg_lld, val_score))
        
        theta, beta = update_theta_beta(train_data, lr, theta, beta, iter)
        #print("Theta: {} \t Beta: {}".format(theta, beta))

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_acc_list ,val_acc_list, train_llds, val_llds


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
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    '''        
    # # Grid Search For Best Hyperparameters
    # lrs = {0.001,0.01,0.1}
    # iterations = {10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,60,80}
    # for lr in lrs:
    #     for iteration in iterations:
    #         theta, beta, train_accs, val_accs, train_llds, val_llds = irt(train_data, val_data, lr, iteration)
    #         print("Final Validation Accuracy (LR={}, IT={}): {}".format(lr,iteration,val_accs[-1]))
    #         print("Final Train Accuracy (LR={}, IT={}): {}".format(lr,iteration,train_accs[-1]))
    '''
    
    # Defining the Best Hyperparameters
    lr = 0.01
    num_iterations = 26
    
    # Training the Final Model
    theta, beta, _, val_accs, train_llds, val_llds = irt(train_data, val_data, lr, num_iterations)
    
    # Obtaining the Final Results
    iterations_list = list(range(1,num_iterations+1))
    final_val_acc = val_accs[-1]
    final_test_acc = evaluate(test_data, theta, beta)
    print("The Final Validation Accuracy is: {}".format(final_val_acc))
    print("The Final Test Accuracy is: {}".format(final_test_acc))
    plt.title("Negative Log Likelihood vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Negative Log Likelihood")
    plt.xticks(iterations_list)
    plt.plot(iterations_list,train_llds,label = "Train")
    plt.plot(iterations_list,val_llds,label = 'Validation')
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)   
    questions = [8,88,888]
    
    prob_correct_1 = []
    for user in range(len(theta)):
        prob_correct_1.append(sigmoid(theta[user]-beta[questions[0]]))
    
    prob_correct_2 = []
    for user in range(len(theta)):
        prob_correct_2.append(sigmoid(theta[user]-beta[questions[1]]))
        
    prob_correct_3 = []
    for user in range(len(theta)):
        prob_correct_3.append(sigmoid(theta[user]-beta[questions[2]]))
    
    plt.title("Probability of Correct vs Theta for Specific Questions")
    plt.scatter(theta,prob_correct_1,s=0.5,label='Question 8')
    plt.scatter(theta,prob_correct_2,s=0.5,label='Question 88')
    plt.scatter(theta,prob_correct_3,s=0.5,label='Question 888')
    plt.legend()
    plt.show()
    
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    
if __name__ == "__main__":
    main()
