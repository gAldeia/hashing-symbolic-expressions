# use this one because it doesnt have hyper-parameters

# Good reference: https://webdocs.cs.ualberta.ca/~games/go/seminar/notes/2007/slides_ucb.pdf

from .BaseLearner import BaseLearner
import numpy as np


class UCB1Learner(BaseLearner):
    def __init__(self, n_obj, n_arms, arm_labels, **kwargs):
        super(UCB1Learner, self).__init__(n_obj=n_obj, n_arms=n_arms, arm_labels=arm_labels)
        
        self._avg_rewards = np.zeros(n_arms)
        self._num_pulls   = np.zeros(n_arms)


    def _calculate_UCB1s(self):
        # We need that the reward is in [0, 1] (not avg_reward, as it seems to
        # render worse results). It looks like normalizing the rewards is a
        # problem: reward should be [0, 1], but not necessarely avg_rewards too
        rs = self._avg_rewards
        ns = self._num_pulls
        
        return rs + np.sqrt(2*np.log1p(sum(ns))/(ns+1))


    def choose_arm(self, context):
        """Uses previous recordings of rewards to pick the arm that maximizes
        the UCB1 function. The choice is made in a deterministic way.
        """
        arm =  np.nanargmax( self._calculate_UCB1s() )

        return self.arm_labels[arm]
    

    def update(self, arm, delta_costs, context):
        reward = self._calc_reward(delta_costs)
        
        self.log(arm, delta_costs, context)

        arm_idx = self.arm_labels.index(arm)

        # Updating counters
        self._num_pulls[arm_idx]   = self._num_pulls[arm_idx] +1
        self._avg_rewards[arm_idx] = self._avg_rewards[arm_idx] + \
            ((reward - self._avg_rewards[arm_idx])/self._num_pulls[arm_idx])