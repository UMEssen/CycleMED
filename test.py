"""Inference script for CycleGAN-based medical image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the parsed options.
It then runs inference for '--num_test' images and saves results to an HTML file.

Use '--model cycle_gan' with the unaligned dataset for bidirectional CycleGAN inference, or
'--model test' with the single dataset for one-direction generator inference. See README.md
for reproducible medical inference and NIfTI reconstruction commands.
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    wandb = None


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        if wandb is None:
            raise ImportError('wandb is required when --use_wandb is enabled.')
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='Medical-CycleGAN')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # Test with eval mode. This only affects layers like batchnorm and dropout.
    # CycleGAN uses instance normalization without dropout by default.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
            use_wandb=opt.use_wandb,
            scale_to_hu=opt.scale_to_hu,
            verbose=opt.verbose,
        )
    webpage.save()  # save the HTML
