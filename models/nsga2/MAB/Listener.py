#Can only be updated, but cannot make any choice. Used to keep track of events only
# (to have access to plot functions)

from .BaseLearner import BaseLearner


class Listener(BaseLearner):
    def __init__(self, n_obj, n_arms, arm_labels, **kwargs):
        super(Listener, self).__init__(
            n_obj=n_obj, n_arms=n_arms, arm_labels=arm_labels)


    def update(self, arm, delta_costs, context):
        self.log(arm, delta_costs, context)
        
        return self