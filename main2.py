import yaml
from reinforcement_learner import ReinfocementLearner

def main():
    rl = ReinfocementLearner(yaml.safe_load(open('config.yaml', 'r')))
    rl.train()
    rl.plot_loss()
    rl.plot_winnings()

if __name__ == '__main__':
    main()