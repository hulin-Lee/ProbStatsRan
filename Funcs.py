# flake8: noqa: E501 W293
import numpy as np
import scipy.optimize

def multi_hypo_test(data, hypo, prior):
    '''
    Perform hypothesis testing on binary data. 
    Binary data means the observed data can only take two values, like good or bad for a product.
    
    Given the observed data, calculate the posterior probabilities for different hypotheses.
    
    Parameters:
    - data: A list of binary values, e.g., [0, 1, 0, 1, 0, 1, 0, 1, 0, 1].
    - hypo: A list of probabilities for each hypothesis being true, e.g., [0.5, 0.5] represents the probability of the data being 0 under each hypothesis.
    - prior: A list of prior probabilities for each hypothesis, e.g., [0.5, 0.5].
    
    Returns:
    - posterior: A list of posterior probabilities for each hypothesis.
    - dB: The calculated decibel values based on the odds of the hypotheses.
    
    Reference: This method is based on the work in Jaynes' "Probability Theory: The Logic of Science", Chapter 4.
    '''
    data_prob = np.ones(len(hypo)) * prior  # Initialize the probabilities of data for each hypothesis based on prior probabilities.
    
    for i in range(len(hypo)):
        for j in data:
            if j == 0:
                data_prob[i] *= hypo[i]  # Update the probability if the data value is 0.
            else:
                data_prob[i] *= (1 - hypo[i])  # Update the probability if the data value is 1.
    
    posterior = np.array([i / np.sum(data_prob) for i in data_prob])  
    
    odds = np.array([posterior[i] / (np.sum(posterior) - posterior[i]) for i in range(len(posterior))])
    
    dB = 10 * np.log10(odds)  
    
    return posterior, dB


def best_strategy(odds):
    '''
    Assume the only types of wagers allowed are to choose one of the outcomes i, i=1,…,m, and bet that i is the outcome of the experiment.
    Use the arbitrage theorem to determine if there is a sure-win betting strategy. 
    If such a strategy exists, solve it using linear programming through the simplex method.

    Args:
    odds: A array of odds for each outcome.

    Returns:
    If a sure-win strategy exists, the function returns the optimal values for each bet and the value of v (profit margin).
    Otherwise, it prints that no sure-win strategy exists.

    Reference:
    This method is based on "Introduction to Probability Models, 11E" by Sheldon M. Ross, Chapter 10. 
    See the example 10.2 for more details.
    '''
    tolerence = 1e-6
    if abs(np.sum(1/(1+odds))-1) > tolerence:  
        print("A sure-win strategy exists")
    else:
        print("No sure-win strategy exists")
        return
    A = np.ones((len(odds), len(odds)+1))  # Constraint matrix
    for i in range(len(odds)):
        A[i][i] = -odds[i] 
    # Objective function: maximize v, so set the coefficient for v to -1 (since linprog minimizes by default)
    c = [0] * len(odds) +[-1]  # The last element is the coefficient for v (we maximize -v)
    b = [0] * len(odds) # Right-hand side constants
    
    x_bounds = [(-1, 1)] * len(odds)  # The bounds for x1, x2, x3
    bounds = x_bounds + [(0, None)]  # v must be greater than zero
    result = scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    if result.success:
        x1, x2, x3, v = result.x
        print(f'Optimal values: x1 = {x1}, x2 = {x2}, x3 = {x3}, v = {-result.fun}')
        return x1, x2, x3, -result.fun
    else:
        print("Optimization failed.")
        print(f"Status: {result.message}")
        print(f"Status code: {result.status}")
        print(f"Objective function value: {result.fun}")

              
def markov_stable_state(x, p, method='past_coupling', x0=None, alpha=None):
    '''
    Generate the state of a stationary Markov chain using two methods: past coupling or an alternative method.
    
    Parameters:
    - x: A list of possible states, e.g., [1, 2, 3, 4].
    - p: A transition matrix where p[i][j] indicates the probability of moving from state i to state j. For example:
      [[0.1, 0.2, 0.3, 0.4], 
       [0.2, 0.3, 0.4, 0.1], 
       [0.3, 0.4, 0.1, 0.2], 
       [0.4, 0.1, 0.2, 0.3]].
    
    - method: 'past_coupling' to use the past coupling method or 'other' for an alternative method (default is 'past_coupling').
    - x0: The initial state (required if using the alternative method).
    - alpha: The event occurrence rate (required if using the alternative method).
    
    Returns:
    - The steady state based on the chosen method.
    
    Reference: This method is based on "Introduction to Probability Models, 11E" by Sheldon M.Ross, Chapter 11.
    '''
    if method == 'past_coupling':
       
        states = np.array([np.random.default_rng().choice(x, p=p[i]) for i in range(4)])
        temp_states = np.array([0, 0, 0, 0])
        
        for i in range(50):  
            N = np.array([np.random.default_rng().choice(x, p=p[j]) for j in range(4)])
            # The states genereated by each person N_{-2}(i), N_{-3}(i), ...
            for k in range(4):
                temp_states[k] = states[N[k] - 1]  # The index of state N is N-1
            states = np.array([temp_states[k] for k in range(4)])
            
            if np.all(states == states[0]): 
                break
            if i == 49: 
                raise ValueError('The Markov chain has not reached a steady state in 50 iterations!')
                
        return states[0] 
    
    elif method == 'other':
       
        states = []
        flag = 0
        states.append(x0)  
        i = 1
        
        while flag == 0:
            state = np.random.default_rng().choice(x, p=p[states[i - 1] - 1])  
            
            if state == x0:
                ran = np.random.default_rng().random()  
                
                if ran < alpha / p[states[i - 1] - 1][3]:  
                    flag = 1  
                    
            states.append(state)  
            
            if flag == 1:
                return states[i - 1]  
                
            i += 1 
    else:
        raise ValueError('Invalid method!') 
