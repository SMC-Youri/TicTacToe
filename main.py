import random
from bke import MLAgent, is_winner, opponent, RandomAgent, validate, plot_validation, train, train_and_plot, train_and_validate
 
class MyAgent(MLAgent):
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1
        elif is_winner(board, opponent[self.symbol]):
            reward = -1
        else:
            reward = 0
        return reward
 
random.seed(1)
 
my_agent = MyAgent(alpha=0.1, epsilon=0.2)
random_agent = RandomAgent()

train_and_validate(
  agent=my_agent,
  validation_agent=random_agent,
  iterations=50,
  trainings=500,
  validations=1000)

my_agent.learning = False

validation_result = validate(    
    agent_x=my_agent,
    agent_o=random_agent,
    iterations=10000,)
 
plot_validation(validation_result)