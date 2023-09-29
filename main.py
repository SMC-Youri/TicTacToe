import random
from bke import MLAgent, is_winner, opponent, RandomAgent, validate, plot_validation, train, train_and_validate

#MyAgent class aangemaakt
class MyAgent(MLAgent):
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1
        elif is_winner(board, opponent[self.symbol]):
            reward = -1
        else:
            reward = 0
        return reward

#Seed toegevoegd zodat deze niet willekeurig is
random.seed(1)

#Hyperparameters toegevoegd aan MyAgent
my_agent = MyAgent(alpha=0.1, epsilon=0.2)
random_agent = RandomAgent()

#MyAgent wordt getraind
train_and_validate(
  agent=my_agent,
  validation_agent=random_agent,
  iterations=50,
  trainings=500,
  validations=1000)

#MyAgent stopt met leren
my_agent.learning = False

#MyAgent wordt gevalidate
validation_result = validate(    
    agent_x=my_agent,
    agent_o=random_agent,
    iterations=10000,)

#Grafiek wordt geplot
plot_validation(validation_result)