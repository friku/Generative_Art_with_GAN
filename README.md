# Generative_Art_with_GAN
The trained GAN can create data similar to the training data, and various expressive abilities are acquired inside the GAN to generate data. We utilized this expressive ability to develop a new expressive method.
The gif image below is an example of a new representation image using GAN. Stained glass images can be obtained by inputting special noises instead of the GAN intermediate representation learned from CelebA images.

Details will be added later.

![sample](https://raw.githubusercontent.com/wiki/friku/Generative_Art_with_GAN/example.gif)



# System requirements

- 64-bit Python 3.6 installation with numpy 1.13.3 or newer. We recommend  Anaconda3.
- NVIDIA driver 385 or newer
- Tensorflow 1.2 or newer
- numpy>=1.13.3
- Pillow>=3.1.1

# Using pre-trained networks

A minimal example of using a pre-trained Generative_Art_with_GAN generator is given in generate_art_minimal.py.

Before executed, you must download a learned model from Google Drive:
https://drive.google.com/open?id=1N9AvyKx6atBy-Ewcmcd3EyVZwXT3xQ-t

Put the downloaded checkpoints in the same directory as generate_art_minimal.py.

The execution command is `python generate_art_minimal.py`

## Training networks

Once the datasets are set up, you can train your own Generative_Art_with_GAN networks as follows:

1. Edit [train_celeba_wgan_gp_pos.py](./train_celeba_wgan_gp_pos.py) to specify the dataset directory configuration by  editing specific lines.
2. Run the training script with `python train_celeba_wgan_gp_pos.py`.
3. The results are written to a newly created directory `./sample_images_while_training/<image name>`.
4. The training may take several hours to complete.





# stylegan
