# Proof of concept - Vision transformer with ARC tasks, and predict a large area, and visualization - Status: Somewhat working

The file `classify_colors2.ipynb` is the best so far.

The file `classify_colors3.ipynb` is work in progress with Lightning and running the trainer from command line.

Mission accomplished: The goal is the output an image with different colored pixels.
Rarely the model outputs an image with a few pixels in another color than the background. With this image it's possible to determine if the model is getting closer to the target image.

Most of the time, the model outputs an image with a single color for all the pixels. These image are useless, it's not possible to tell if the model is improving.
The model predictions are very similar vectors, no matter what x, y coordinate I provide.
So when making predictions for each pixel in the output image. Then all the pixels have the same color.

Lesson: Onehot encoding of the input image. Each input pixel can have 13 values: 0, 1, 2...11, 12.
I tried using the RGB color values in 3 color channels and it outputted an image with 1 single color for all pixels.
It's likely due to the training where I used multiple ARC tasks at the same time, confusing the model. 
I tried using only the Red color channel and zero in the Green+Blue channel and it outputted an image with 1 single color for all pixels.
It could be interesting trying out the other approaches again, since it requires 3 channels, vs. 13 channels that I'm using now.
And 3 channels has a smaller memory footprint than 13 channels.

Lesson: When training, only train using 1 ARC task at a time.
I made the mistake while training, using around 6 ARC tasks at a time, causing the weights to be an average of the 6 tasks. And it took a long time to improve on its accuracy.
That's my hypothesis why it behaved so poorly. I was only seeing 1 embedding vector for all the output pixels.

I use ViT to do image classification, and determine which one of 10 classes to pick.
The input image has a marker that indicates the pixel that is to be predicted.

I hope can a ViT do a better job than all the other approaches I have been trying out.

My logistic regression code solves 51 tasks out of 800 tasks, but zero on the hidden ARC dataset.
The shortcoming of the logistic regression is that it doesn't learn stuff. It starts from scratch every time.
The logistic regression is provided with several augmentations.

I can't provide all the augmentations to the ViT, since it will take forever to do training.
And my augmentations is something I always make changes to, so it's fragile to base a model on the augmentations.
So if I can avoid my augmentations, then it will be preferred.

With LODA-RUST I have generated several megabytes of data for a few ARC tasks.
I use only the Red channel assigned with the ARC symbol, in the range 0..255.
The Green + Blue channels are unused.
The size of the images varies.
Each image has the size 30x30. 

I have tried generating images that all have the size 224x224, which seems to be
what the ViT example Dogs&Cats are using. But I don't see any gain in accuracy.
So I provide the images in varying sizes.

## Stats A - ARC

This is with 1 ARC tasks: 10 epochs takes 10m on a M1 Mac. 

Each task have 10.000 images in the `train` dir. And there are 10 classes: color0..color9.
The `tasks.zip` is 84mb. The images are highly similar.

Training with dataset:
```
Epoch : 118 - loss : 0.8183 - acc: 0.6970 - val_loss : 0.7954 - val_acc: 0.7046
Epoch : 119 - loss : 0.8180 - acc: 0.6959 - val_loss : 0.7950 - val_acc: 0.7109
Epoch : 120 - loss : 0.8180 - acc: 0.6982 - val_loss : 0.7956 - val_acc: 0.7002
```


## Stats B - Cats & Dogs

There are around 45.000 images in the `train` dir, and only 2 classes: cat, dog.
The `train.zip` is 570mb. The `test.zip` is 284mb. Every image is unique.

```
Epoch : 1 - loss : 0.6925 - acc: 0.5198 - val_loss : 0.6803 - val_acc: 0.5775
Epoch : 2 - loss : 0.6829 - acc: 0.5500 - val_loss : 0.6627 - val_acc: 0.5975
Epoch : 3 - loss : 0.6692 - acc: 0.5754 - val_loss : 0.6468 - val_acc: 0.6226
…
Epoch : 18 - loss : 0.5959 - acc: 0.6779 - val_loss : 0.5625 - val_acc: 0.7104
Epoch : 19 - loss : 0.5933 - acc: 0.6791 - val_loss : 0.5658 - val_acc: 0.7061
Epoch : 20 - loss : 0.5919 - acc: 0.6805 - val_loss : 0.5659 - val_acc: 0.7059
```

## Comparing stats A vs. stats B

In `Cats & Dogs`, the `val_acc` is `0.7059` after only 20 epochs. This is very impressive.

In ARC the `val_acc` is `0.7002` after 120 epochs. This is good.
Ways to improve the `val_acc`.
The size of the ARC data is much smaller than the Cats & Dogs data.
If I generate even more permutations, to get to the same size.
Since ARC has more classifications than the cats & dogs, it may require even more training data to get to a comparable accuracy.

## Future plans

More data. 
- Add more tasks.
- Permute the existing tasks even more.
- Varying sizes, so sometimes 30x30 is used, other times a compact representation is used.

Maybe integrate with HuggingFace's "Trainer".
https://huggingface.co/docs/transformers/main_classes/trainer
I couldn't get it to work with mps, and it seems lots of other people are also experiencing problems with mps,
so I doubt that mps support is sufficiently robust.
