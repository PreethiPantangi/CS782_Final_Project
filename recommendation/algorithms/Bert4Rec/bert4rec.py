import torch

from recommendation.algorithms.Bert4Rec.runoptions import args
from recommendation.algorithms.Bert4Rec.models import model_factory
from recommendation.algorithms.Bert4Rec.dataloaders import dataloader_factory
from recommendation.algorithms.Bert4Rec.trainers import trainer_factory
from recommendation.algorithms.Bert4Rec.utils import *
import sys
import os

# Now you should be able to import the 'options' module
from recommendation.algorithms.Bert4Rec.runoptions import args


def train(args):
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    # test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    # if test_model:
    trainer.test()


# if __name__ == '__main__':
#     if args.mode == 'train':
#         train()
#     else:
#         raise ValueError('Invalid mode')
