# Describes the basic interface for the learners
import numpy as np
import pandas as pd

class BaseLearner:
    # Learners will use parameters's default values inside brush Hacked.
    # Defalt values in __init__ function of learners should be compatible
    # with brush experiments. each main function in the learner's files should
    # work with these default values as well.
    def __init__(self, n_obj, n_arms, arm_labels, weights=None):

        self.n_obj  = n_obj
        self.n_arms = n_arms

        # standard (minimum) information about state to be logged.
        self.pull_history = {
            c:[] for c in ['t', 'arm', 'reward', 'update', 'delta_error', 'gen']}
        
        # Initializing non-default arguments

        # Weights --------------------------------------------------------------
        if weights is None: # Minimizing objectives by default
            weights = np.array([-1.0 for _ in range(self.n_obj)])

        assert len(weights) == n_obj, \
            "number of weights must match number of objectives"
        
        # make sure it can be broadcasted in operations
        self.weights = np.array(weights)

        # arm labels -----------------------------------------------------------
        if arm_labels is None or len(arm_labels) != self.n_arms:
            arm_labels = [f'arm {i}' for i in range(self.n_arms)]

        self.arm_labels = arm_labels


    @property
    def probabilities(self):
        raise NotImplementedError()
    
    
    @probabilities.setter
    def probabilities(self, new_probabilities):
        raise NotImplementedError()


    def choose_arm(self, **context):
        raise NotImplementedError()
    

    # Methods to calculate the pareto dominance, used to define the reward
    def _epsi_less(self, a, b, e):
        return a < b and not self._epsi_eq(a, b, e)
    

    def _epsi_eq(self, a, b, e):
        return np.abs(a-b) <= e
    

    def _epsi_lesseq(self, a, b, e):
        return self._epsi_less(a, b, e) or self._epsi_eq(a, b, e)


    def _epsi_dominates(self, a, b, eps=None):
        # Returns true if a DOMINATES b. We consider dominance as a minimization problem by default
        # a needs to be <= for all values in b, and strictly < for at least one value.
        # The weights must be +1 for maximization problems and -1 for minimization.

        if eps is None:
            eps = [1e-5 for _ in zip(a, b)]

        return all(self._epsi_lesseq(ai, bi, ei) for ai, bi, ei in zip(a, b, eps)) \
        and    any(self._epsi_less(ai, bi, ei)   for ai, bi, ei in zip(a, b, eps))
    

    def _calc_reward(self, delta_costs, eps=None):
        #delta_costs = np.subtract(ind1.fitness.values, offspring.fitness.values)

        delta_costs = np.array(delta_costs)
        # `delta_costs` will be a numpy array with multiple values (deltas for
        # each objective function). Each learner needs to figure out how it will
        # handle the values.

        # Example how to decide if the reward should be positive
        # At least one thing got better and the others didn't got worse
        # Rewards are always expected to be an array (even when it is one value)
        reward = 1.0 if (delta_costs[0]<0) else 0
        return reward

        if reward == 0 and delta_costs[0]< 1e-6:
            return 0.5
        return reward
    
        # dominance based reward
        if eps is None:
            eps = [0.0 for _ in delta_costs] # one for each objective

        reward = 0.0 # in case none of if statements below work, this means offspring is dominated by the parent
        # if reward=1, then self._epsi_dominates(self.weights*delta_costs, np.zeros_like(delta_costs), eps, 
        # and A is non dominated by B (improved at least one objective)

        # proportional to number of objectives where offspring is not dominated
        # self._epsi_dominates(np.zeros_like(delta_costs), self.weights*delta_costs):# b dominates a (improved all objectives)

        num_improvements = sum(
            [1.0 if self._epsi_less(0, ai, ei) else 0.0
             for ai, ei in zip(self.weights*delta_costs, eps)])

        reward = np.floor(num_improvements/len(delta_costs))

        #reward = num_improvements/len(self.weights) # Scale accordingly to how many objectives it improved
        
        # Flip a coin in case of a tie
        # s = num_improvements/len(delta_costs)
        # reward = np.random.choice([0, 1], p=[1-s, s])

        # reward =  1.0 if reward>0 else 0.0 # Positive reward if it improved at least one objective
        
        # reward based on pareto dominance --- offsprint is not dominated (weak version)
        # reward = 1.0 if not self._epsi_dominates(
        #     self.weights*delta_costs,
        #     np.zeros_like(delta_costs),
        #     eps) else 0.0

        # Improves one obj without decreasing the other -------------------------
        # reward = 0
        # if (self.weights[0]*delta_costs[0]<0 and self.weights[1]*delta_costs[1]<=0
        # or  self.weights[1]*delta_costs[1]<0 and self.weights[0]*delta_costs[0]<=0):   
        #     reward = 1

        return reward
    

    def log(self, arm, delta_costs, context, reward=None):
        # Should be called inside update, or when a learner fails but still
        # need to report what happened. If the log needs more information,
        # then it should be handled by the sub-class.

        if reward is None:
            reward = self._calc_reward(delta_costs)

        for k, v in [
            ('t',           len(self.pull_history['t']) ),
            ('arm',         arm                         ),
            ('reward',      reward                      ),
            ('delta_error', self.weights*delta_costs    ),
            ('gen',         context.get('gen', None)    ),
            ('update',      0                           ),
        ]:
            self.pull_history[k].append(v)

        return self


    def update(self, arm, delta_costs, context):
        raise NotImplementedError()
    

    def calculate_statistics(self):

        # Turning it into a pandas dataframe so we can easily calculate stats
        learner_log = pd.DataFrame(self.pull_history).set_index('t')
        
        # TODO: rename it to total_non-empty_rewards (or something similar), because reward can be 0.5
        total_rewards = {i:0 for i in range(self.n_arms)}
        total_rewards.update(learner_log[learner_log['reward']>0.0].groupby('arm')['reward'].size().to_dict())

        total_half_rewards = {i:0 for i in range(self.n_arms)}
        total_half_rewards.update(learner_log[(learner_log['reward']<1.0) & (learner_log['reward']>0.0)
                                              ].groupby('arm')['reward'].size().to_dict())

        # avoiding have a different number of values in dict when one or more arms are not used
        total_pulls = {i:0 for i in range(self.n_arms)}
        total_pulls.update(learner_log['arm'].value_counts().to_dict())

        # We use sorted because they are all dicts, so ordering is the same 
        data_total_pulls        = np.array([total_pulls[k] for k in sorted(total_pulls)])
        data_total_rewards      = np.array([total_rewards[k] for k in sorted(total_rewards)])
        data_total_half_rewards = np.array([total_half_rewards[k] for k in sorted(total_half_rewards)])

        statistics = pd.DataFrame.from_dict({
            'arm'         : self.arm_labels,
            'totpulls'    : data_total_pulls,
            'pulls%'      : np.nan_to_num( (data_total_pulls/data_total_pulls.sum()).round(2) ),
            '+reward'     : data_total_rewards,
            'part. reward': data_total_half_rewards,
        })

        return statistics