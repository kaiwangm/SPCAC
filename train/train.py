import argparse
import warnings

from train.trainer import create_trainer

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

# -------------------- argument -------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='default')
parser.add_argument('--model', type=str, default='baseline_factorized')
parser.add_argument('--quality', type=int, default=0)
parser.add_argument('--dataset', type=str, default=None,
                    help='Dataset name (overrides the trainer config default)')
opt = parser.parse_args()


def main():
    """ Main function to train the model."""
    print('----------------------------')
    print(f"Training {opt.model} model with dataset {opt.dataset or 'config default'}")
    print('----------------------------')
    model_trainer = create_trainer(opt.config, opt.model, opt.quality, opt.dataset)
    model_trainer.train()
    print('----------------------------')
    print('Training finished')


if __name__ == "__main__":
    main()
