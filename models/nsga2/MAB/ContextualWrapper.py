
from .BaseLearner import BaseLearner
import numpy as np


class Ball():
    def __init__(self, center, radius, Learner, Learner_kwargs, id):
        self.center = center
        self.radius = radius

        self.counter = 0
        self.learner = Learner(**Learner_kwargs)

        self.id     = id # identifier (by the order the balls are created)
        self.active = True


    def choose_arm(self, context):
        return self.learner.choose_arm(context)


    def update(self, arm, delta_costs, context):
        self.learner.update(arm, delta_costs, context)
        self.counter = self.counter + 1


class ContextSpace():
    def __init__(self, *arrays):
        # our context set is discretized by the cartesian product of elements in `arrays`

        if len(arrays) == 0:
            raise ValueError("At least one array required as input")
        
        max_distance = [ 
            np.array([np.max(d) for d in arrays]) - 
            np.array([np.min(d) for d in arrays])
        ]

        def d(x, xprime):
            # x and xprime must be of same size than number of contexts

            dist = np.linalg.norm(
                    (self._get_context(x) - self._get_context(xprime)) \
                    / max_distance )
                
            # Bounding the distance so the diameter of context space is 1
            return np.minimum( 1.0, dist )

        self.n_dimensions = len(arrays)
        self.dimensions   = arrays
        self.d            = d

    def _get_context(self, x): # return the context partition indexes
        return np.array([np.searchsorted(d_i, x_i)
                         for (d_i, x_i) in zip(self.dimensions, x)])
    

class ContextualWrapper(BaseLearner):
    def __init__(self, n_obj, n_arms, arm_labels, rnd_generator,        
                 context_keys, context_space, delete_at,
                 Learner, Learner_kwargs={}, **kwargs):
        
        super(ContextualWrapper, self).__init__(
            n_obj=n_obj, n_arms=n_arms, arm_labels=arm_labels)
        
        self.rng_generator  = rnd_generator
        self.context_keys   = context_keys
        self.Learner        = Learner
        self.context_space  = context_space
        self.Learner_kwargs = {**Learner_kwargs,
                               **{'n_obj' : n_obj, 'n_arms' : n_arms, 
                                  'arm_labels' : arm_labels}}
        self.pull_history   = {**self.pull_history, **{'ball_id' : []}}

        # Collection of balls in the context space, initialized with one ball that covers the whole context set
        self.A = [
            Ball(center=np.array([d[(len(d)-1)//2] for d in self.context_space.dimensions]),
                 radius=2.0,
                 Learner=self.Learner,
                 Learner_kwargs=self.Learner_kwargs, 
                 id=1)
        ]
        
        # Set of active balls that are not full
        self.Astar = [ self.A[0] ]
        self.T_0 = lambda r: delete_at

        self.update_queue = []


    def choose_arm(self, context):
        x = [context[k] for k in self.context_keys]
        
        # find set of relevants. if not empty, use a random one
        # else, create new ball
        relevant = [B for B in self.Astar if self.context_space.d(x, B.center)<B.radius]
        
        B = None
        if len(relevant)>0:
            B = self.rng_generator.choice(relevant)
        else:
            # the minimum radius value between all balls that contained x_t in the history of the learner
            r = np.min([B.radius for B in self.A if self.context_space.d(x, B.center)<B.radius])
            B = Ball(
                np.array(x), #self.context_space._get_context_set(x), 
                r/2, 
                Learner=self.Learner, 
                Learner_kwargs=self.Learner_kwargs, 
                id=len(self.A)+1)
            
            self.A.append(B)
            self.Astar.append(B)

        self.update_queue.append(B)
        arm = B.choose_arm(context)
        
        return arm

    def log(self, arm, delta_costs, context):

        assert len(self.update_queue)>0, "trying to update without pulling before"

        self.last_B_ = self.update_queue[0] 
        self.update_queue = self.update_queue[1:]

        # It's not used to make decisions, but we still need to keep track
        reward = self.last_B_.learner._calc_reward(delta_costs)

        super().log(arm, delta_costs, context, reward)

        self.pull_history['ball_id'].append( self.last_B_.id )


    def update(self, arm, delta_costs, context):
        
        # Log will check for an empty queue and also update the last_B_ used here
        self.log(arm, delta_costs, context)

        # Update learner
        self.last_B_.update(arm, delta_costs, context)

        # update ball
        self.last_B_.counter = self.last_B_.counter + 1

        # remove ball if necessary
        if self.last_B_.counter >= self.T_0(self.last_B_.radius):
            self.last_B_.active = False
            self.Astar.remove(self.last_B_)