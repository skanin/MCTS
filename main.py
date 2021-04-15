import yaml
from reinforcement_learner import ReinfocementLearner
from topp import Topp

def train_model(cfg):
    rl = ReinfocementLearner(cfg)
    rl.train()
    rl.plot_loss()
    rl.plot_winnings()


def topp(cfg):
    toppHex = Topp(cfg)
    toppHex.play_tournament()

if __name__ == '__main__':
    # train_model(yaml.safe_load(open('config.yaml', 'r')))

    topp(yaml.safe_load(open('toppcfg.yaml', 'r')))
