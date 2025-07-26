from abc import ABCMeta, abstractmethod
import random
import numpy as np
from gym.spaces import Dict, MultiBinary
import gym
import copy
import math
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager

# Constants
MENTS_FACTOR = 10  # Number of iterations for MENTS
ROLLOUT_FACTOR = 3  # Number of rollouts for estimating rewards

class BaseAgent(metaclass=ABCMeta):

    @abstractmethod
    def sample_action(self, state, env):
        pass

class RandomAgent(BaseAgent):
    def __init__(self, action_space, num_actions=1, seed=None):
        self.action_space = action_space
        self.num_actions = num_actions
        if seed is not None:
            self.action_space.seed(seed)

    def sample_action(self, env=None):
        s = self.action_space.sample()
        action = {}
        selected_actions = random.sample(list(s), self.num_actions)
        for sample in selected_actions:
            if isinstance(self.action_space[sample], gym.spaces.Box):
                action[sample] = s[sample][0].item()
            elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                action[sample] = s[sample]
        return action

class NoOpAgent(BaseAgent):
    def __init__(self, action_space, num_actions=0):
        self.action_space = action_space
        self.num_actions = num_actions

    def sample_action(self, state=None):
        action = {}
        return action

class MENTSAgent(BaseAgent):

    def __init__(self, action_space, num_actions=1, seed=None):
        self.action_space = action_space
        self.num_actions = num_actions
        if seed is not None:
            self.action_space.seed(seed)

    def sample_action(self, env):
        s = self._get_action_with_MENTS(env)
        action = {}
        selected_actions = random.sample(list(s), self.num_actions)
        for sample in selected_actions:
            if isinstance(self.action_space[sample], gym.spaces.Box):
                action[sample] = s[sample][0].item()
            elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                action[sample] = s[sample]
        return action
    
    def _get_action_with_MENTS(self, env):
        root = MENTSNode(env=env, action_space=self.action_space)
        return root.best_action()

class MENTSNode():
    def __init__(self, env, action_space, parent=None, parent_action=None):
        self.env = env
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.number_of_visits = 0
        self.reward_distribution = []  # Stores rewards for entropy calculation
        self.untried_actions = [0, 1]  # Example actions (should be dynamic based on the environment)
        self.action_space = action_space
        self.total_outcomes = 0
        # self.outcome_counts = {0: 0, 1: 0}
        self.outcome_counts = {}


    def expand(self):
        action = self.untried_actions.pop()
        EnvInfo = ExampleManager.GetEnvInfo("Traffic")
        new_env = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), 
                                  instance=EnvInfo.get_instance(0))
        new_env.sampler.subs = copy.deepcopy(self.env.sampler.subs)
        new_env.state = copy.deepcopy(self.env.state)

        next_state, reward, done, info = new_env.step({'advance___i0': action})
        child_node = MENTSNode(env=new_env, action_space=self.action_space, 
                               parent=self, parent_action={'advance___i0': action})
        self.children.append(child_node)
        return child_node

    def rollout(self):
        EnvInfo = ExampleManager.GetEnvInfo("Traffic")
        current_env = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), 
                                      instance=EnvInfo.get_instance(0))
        current_env.sampler.subs = copy.deepcopy(self.env.sampler.subs)
        current_env.state = copy.deepcopy(self.env.state)

        total_reward = 0
        for i in range(ROLLOUT_FACTOR):
            action = self.rollout_policy(self.action_space)
            current_rollout_state, reward, done, info = current_env.step(action)
            total_reward += reward
        return total_reward

    # def backpropagate(self, result):
    #     self.number_of_visits += 1
    #     self.reward_distribution.append(result)
    #     if self.parent:
    #         self.parent.backpropagate(result)

    def backpropagate(self, result):
        self.number_of_visits += 1
        self.reward_distribution.append(result)
        # Update total_outcomes and outcome_counts
        self.total_outcomes += 1
        if result not in self.outcome_counts:
            self.outcome_counts[result] = 0
        self.outcome_counts[result] += 1
        if self.parent:
            self.parent.backpropagate(result)


    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self):
        choices_weights = []
        for child in self.children:
            if child.number_of_visits == 0:
                return child
            else :
                choices_weights.append(child.calculate_entropy())
        return self.children[np.argmax(choices_weights)]

    # def calculate_entropy(self, rewards):
    #     if not rewards:
    #         return 0
    #     probabilities = np.array(rewards) / np.sum(rewards)
    #     return -np.sum(probabilities * np.log(probabilities + 1e-9))
    
    # def calculate_entropy(self):
    #     if self.total_outcomes == 0:
    #         return float('inf')  # Favor unvisited nodes
    #     entropy = 0
    #     for outcome, count in self.outcome_counts.items():
    #         p = count / self.total_outcomes
    #         entropy -= p * math.log(p)
    #     return entropy
    
    def calculate_entropy(self):
        if self.total_outcomes == 0:
            return float('inf')  # Favor unvisited nodes
        entropy = 0
        for outcome, count in self.outcome_counts.items():
            p = count / self.total_outcomes
            entropy -= p * math.log(p + 1e-9)  # Add a small value to avoid log(0)
        return entropy

    
    def rollout_policy(self, action_space):
        return action_space.sample()

    def tree_policy(self):
        current_node = self
        while True:
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        # return current_node

    def best_action(self):
        for i in range(MENTS_FACTOR):
            v = self.tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.best_child().parent_action