import numpy as np
import matplotlib.pyplot as plt

# – Compute a regret-matching strategy profile. (If all regrets for a player are non-positive, use
#         a uniform random strategy.)
# – Add the strategy profile to the strategy profile sum.
# – Select each player action profile according the strategy profile.
# – Compute player regrets.
# – Add player regrets to player cumulative regrets.

class RPS():
    def __init__(self, n_actions, opponent_strat, utility):
        self.opponent_strat = opponent_strat
        self.n_actions = n_actions
        self.utility = utility

        self.regret_sum = np.array([0,0,0], dtype=np.float32)
        self.strat_sum = np.array([0,0,0], dtype=np.float32)


    def get_strategy(self):
        strat = np.maximum(self.regret_sum, 0)
        if sum(strat) > 0:
            strat = strat / sum(strat)
        else:
            strat = [1.0 / self.n_actions]*self.n_actions
        self.strat_sum += strat

        return strat

    def get_action(self, strat):
        return np.random.choice(self.n_actions, 1, p=strat)

    def train(self, n_iters):
        for i in range(n_iters):
            # get regret-matched mixed-strategy actions
            strat = self.get_strategy()
            my_action = self.get_action(strat)
            other_action = self.get_action(self.opponent_strat)

            # accumulate action regrets
            for regret_action in range(self.n_actions):
                self.regret_sum[regret_action] += self.utility[regret_action, other_action] - self.utility[my_action, other_action]
            print(self.regret_sum)

    def get_average_strat(self):
        return self.strat_sum / sum(self.strat_sum)




opponent_strat = np.array([0.3, 0.3, 0.4])
n_actions = 3
# in order R P S, me - you
utility = np.array([[0, -1, 1],
                 [1, 0, -1],
                 [-1, 1, 0]], dtype=np.float32)

game = RPS(n_actions, opponent_strat, utility)
game.train(10000)
my_strat = game.get_average_strat()
print('opponent strategy:', opponent_strat)
print('my computed stategy:', my_strat)

opponent_play = np.random.choice(3, 1000, p=opponent_strat) 
my_play = np.random.choice(3, 1000, p=my_strat) 

score = 0
n_play = 1000
for i in range(n_play):
    score += utility[my_play[i], opponent_play[i]]
print('sample reward:', score / n_play)

expected_score = 0
for my_action in range(n_actions):
    for other_action in range(n_actions):
        expected_score += my_strat[my_action]*opponent_strat[other_action]*utility[my_action, other_action]
print('expected reward:', expected_score)
