import gymnasium as gym
from typing import Tuple

import numpy as np
from interfaces.policy import Policy

from assignments.policy_deterministic_greedy import Policy_DeterministicGreedy

def value_prediction(
    env: gym.Env, 
    pi: Policy,
    initV: np.array,
    theta: float,
    gamma: float
) -> Tuple[np.array, np.array]:
    """
    Runs the value prediction algorithm to estimate the value function for a given policy.

    Sutton & Barto, p. 75, "Value Prediction"
    
    Parameters:
        env (gym.Env): environment with model information, i.e. you know transition dynamics and reward function
        pi (Policy): The policy to evaluate (behavior policy)
        initV (np.ndarray): Initial V(s); numpy array shape of [nS,]
        theta (float): The exit criteria
    Returns:
        tuple: A tuple containing:
            - V (np.ndarray): V_pi function; numpy array shape of [nS]
            - Q (np.ndarray): Q_pi function; numpy array shape of [nS,nA]
    """
    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    # Hint: To get the action probability, use pi.action_prob(state,action)
    # Hint: Use the "env.P" to get the transition probabilities.
    #    env.P[state][action] returns a list of tuples [(prob, next_state, reward, done)]
    #    (Both our custom environments and OpenAI Gym environments have this attribute)
    #####################

    P = env.P
    """Transition Dynamics;  env.P[state][action] returns a list of tuples [(prob, next_state, reward, done)]"""
    states = env.observation_space.n
    """Number of states"""
    actions = env.action_space.n
    """Number of actions"""
    V = initV
    """The V(s) function to estimate"""
    Q = np.zeros((states, actions))
    """The Q(s, a) function to estimate"""
    # Policy evaluation part
    while True:
        delta = 0
        newV = V.copy()
        for state in range(states):
            v = newV[state]
         
            newV[state] = 0 # clear the stored cell value
            for action in range(actions):
                action_prob = pi.action_prob(state, action) # Return the probabality of using action in policy
                for prob, next_state, reward, done in P[state][action]:
                    newV[state] += action_prob * prob * (reward + gamma * V[next_state] * (not done))

            # Calculate whether the change in the variable before and after the update has narrowed down to below the threshold.（for all states)
            delta = max(delta, abs(v - newV[state]))
            #Update the value for all states
        V=newV
        if delta < theta:
            break
    
    #above we already updated V,now we calculate Q(s,a) function, using converged V to calculate Q

    for state in range(states):
        for action in range(actions):
            stateQvalue = 0
            for prob, next_state, reward, done in P[state][action]:
                stateQvalue += prob * (reward + gamma * V[next_state] * (not done))
                Q[state][action] = stateQvalue    

    return V, Q


def value_iteration(env: gym.Env, initV: np.ndarray, theta: float, gamma: float) -> Tuple[np.array, Policy]:
    """
    Parameters:
        env (EnvWithModel): environment with model information, i.e. you know transition dynamics and reward function
        initV (np.ndarray): initial V(s); numpy array shape of [nS,]
        theta (float): exit criteria

    Returns:
        tuple: A tuple containing:
            - value (np.ndarray): optimal value function; shape of [nS]
            - policy (GreedyQPolicy): optimal deterministic policy
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    # Hint: Use the "env.P" to get the transition probabilities.
    #    env.P[state][action] returns a list of tuples [(prob, next_state, reward, done)]
    #    (Both our custom environments and OpenAI Gym environments have this attribute)
    # Hint: Try updating the Q function in the `pi` policy object
    #####################
    nS: int = env.observation_space.n
    """Number of states"""
    nA: int = env.action_space.n
    """Number of actions"""
    V: np.ndarray = initV
    """Initial V values"""
    Q: np.ndarray = np.zeros((nS, nA))
    """Initial Q values"""
    pi: Policy_DeterministicGreedy = Policy_DeterministicGreedy(Q)
    """Initial policy, you will need to update this policy after each iteration"""
    P: np.ndarray = env.P
    """Transition Dynamics;  env.P[state][action] returns a list of tuples [(prob, next_state, reward, done)]"""
    while True:
        delta = 0
        newV = V.copy() # use two array to update state values
        for state in range(nS):
            v = newV[state]
            newV[state] = 0
            updated_state_v = max(
                sum(
                    prob * (reward + gamma * V[next_state] * (not done)) for prob, next_state, reward, done in P[state][action]
                ) for action in range(nA)
            )
            delta = max(delta, abs(v - updated_state_v))
            newV[state] = updated_state_v
        V=newV #using updated v-value array to replace the old value
        if delta < theta:
            break

    # Create optimal policy based on the final V(s)
    # Greedy policy based on the optimal Q-values
    for state in range(nS):
        for action in range(nA):
            Q[state][action] = sum(
                prob * (reward + gamma * V[next_state] * (not done))
                for prob, next_state, reward, done in P[state][action]
            )

    return V, Policy_DeterministicGreedy(Q)
