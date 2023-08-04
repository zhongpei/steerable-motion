# Steerable Motion: Fine-tuning Stable Diffusion on Motion Data

Steerable Motion aims to fine-tune Stable Diffusion on Motion data in order to generate motion key frames from a single image.

We believe that Steerable Motion will be a powerful tool for artists who want to create works that match their imagination exactly. The key frames they create will be interpolated into a smooth motion - by both linear and creative interpolation models.

The early versions are built InstructPix2Pix and trained on stills from a single property in order to validate the approach.

You can see some samples from it here:

<img src='https://banodoco.s3.amazonaws.com/examples-small.png'/>

And a **very** early example of what this looks like animated here:

<img src='https://banodoco.s3.amazonaws.com/animated.gif'/>


### Collaborate on this project

We're looking for ML engineers, fine-tuners, artists and hackers to collaborate on this project. If you're interested, you can join our [Discord](https://discord.gg/BkeXnRPyDz) here 


### Test the model

```
conda env create -f environment.yaml
conda activate steerable-motion

```

To test it, you can download the checkpoints from HuggingFace, and use the instructions below to run the code.


```
python edit_cli.py --input input.jpg --output output.jpg --edit "she smiles happily"

# python edit_cli.py --steps 100 --resolution 512 --seed 1371 --cfg-text 7.5 --cfg-image 1.2 --input input.jpg --output output.jpg --edit "she smiles happily"

```

### Train the model

If you'd like to train the model, you can use the following command to download the checkpoints:

```
bash scripts/download_pretrained_sd.sh

```

Then, you can use the following command to train the model:

```

python main.py --name default --base configs/train.yaml --train --gpus 0,1

```
