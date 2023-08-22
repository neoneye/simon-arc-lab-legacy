# Proof of concept - Vision transformer with ARC tasks - Status: Works

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

This is with 3 ARC tasks: 20 epochs takes around 15 minutes on a M1 Mac. 

Each task have 10.000 images in the `train` dir. And there are 10 classes: color0..color9.
The `tasks.zip` is 46mb. The images are highly similar.

```
Epoch : 1 - loss : 2.2896 - acc: 0.1311 - val_loss : 2.2857 - val_acc: 0.1389
Epoch : 2 - loss : 2.2860 - acc: 0.1307 - val_loss : 2.2817 - val_acc: 0.1409
Epoch : 3 - loss : 2.2851 - acc: 0.1378 - val_loss : 2.2809 - val_acc: 0.1501
…
Epoch : 18 - loss : 2.1636 - acc: 0.2141 - val_loss : 2.1331 - val_acc: 0.2130
Epoch : 19 - loss : 2.1548 - acc: 0.2164 - val_loss : 2.1204 - val_acc: 0.2166
Epoch : 20 - loss : 2.1518 - acc: 0.2199 - val_loss : 2.1171 - val_acc: 0.2320
Epoch : 21 - loss : 2.1469 - acc: 0.2232 - val_loss : 2.1074 - val_acc: 0.2232
Epoch : 22 - loss : 2.1374 - acc: 0.2261 - val_loss : 2.1013 - val_acc: 0.2078
Epoch : 23 - loss : 2.1311 - acc: 0.2329 - val_loss : 2.0974 - val_acc: 0.2321
…
Epoch : 38 - loss : 2.0378 - acc: 0.2629 - val_loss : 2.0425 - val_acc: 0.2596
Epoch : 39 - loss : 2.0337 - acc: 0.2662 - val_loss : 2.0123 - val_acc: 0.2704
Epoch : 40 - loss : 2.0250 - acc: 0.2686 - val_loss : 2.0105 - val_acc: 0.2664
Epoch : 41 - loss : 2.0217 - acc: 0.2698 - val_loss : 2.0053 - val_acc: 0.2645
Epoch : 42 - loss : 2.0093 - acc: 0.2760 - val_loss : 1.9945 - val_acc: 0.2592
Epoch : 43 - loss : 2.0062 - acc: 0.2738 - val_loss : 1.9763 - val_acc: 0.2800
…
Epoch : 58 - loss : 1.9168 - acc: 0.3068 - val_loss : 1.9105 - val_acc: 0.2951
Epoch : 59 - loss : 1.9230 - acc: 0.3034 - val_loss : 1.9013 - val_acc: 0.3012
Epoch : 60 - loss : 1.9152 - acc: 0.3061 - val_loss : 1.9323 - val_acc: 0.2912
Epoch : 61 - loss : 1.9101 - acc: 0.3049 - val_loss : 1.8964 - val_acc: 0.3079
Epoch : 62 - loss : 1.8940 - acc: 0.3175 - val_loss : 1.9104 - val_acc: 0.3074
Epoch : 63 - loss : 1.8971 - acc: 0.3132 - val_loss : 1.8935 - val_acc: 0.3074
…
Epoch : 78 - loss : 1.8284 - acc: 0.3347 - val_loss : 1.8795 - val_acc: 0.2953
Epoch : 79 - loss : 1.8242 - acc: 0.3368 - val_loss : 1.8674 - val_acc: 0.3297
Epoch : 80 - loss : 1.8218 - acc: 0.3362 - val_loss : 1.9040 - val_acc: 0.2917
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

In ARC the `val_acc` is `0.2917` after 80 epochs. This is not good.
Ways to improve the `val_acc`.
The size of the ARC data is much smaller than the Cats & Dogs data.
If I generate even more permutations, to get to the same size.
Since ARC has more classifications than the cats & dogs, it may require even more training data to get to a comparable accuracy.

## Future plans

Integrate with HuggingFace's "Trainer".
https://huggingface.co/docs/transformers/main_classes/trainer

More data. 
- Currently only 3 tasks. Add more tasks.
- Permute the existing tasks even more.

Visualize the predicted output.


