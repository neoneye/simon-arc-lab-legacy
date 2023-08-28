# Proof of concept - Vision transformer with ARC tasks, and predict a large area, and visualization - Status: Not working

The model predictions are all the same vector, no matter what x, y coordinate I provide.
So when making predictions for each pixel in the output image. Then all the pixels have the same color.
Something is not working.
The goal is the output an image with different colored pixels.

I have been training the model for 24 hours, but it still continues to output a single color.

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

This is with 6 ARC tasks: 500 epochs takes 23h30m on a M1 Mac. 

Each task have 10.000 images in the `train` dir. And there are 10 classes: color0..color9.
The `tasks.zip` is 84mb. The images are highly similar.

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
Epoch : 272 - loss : 1.1878 - acc: 0.4903 - val_loss : 1.6486 - val_acc: 0.4164
Epoch : 273 - loss : 1.1903 - acc: 0.4896 - val_loss : 1.7267 - val_acc: 0.4072
…
Epoch : 316 - loss : 1.2112 - acc: 0.4846 - val_loss : 1.6958 - val_acc: 0.4112
Epoch : 317 - loss : 1.2096 - acc: 0.4837 - val_loss : 1.6954 - val_acc: 0.4035
Epoch : 318 - loss : 1.2045 - acc: 0.4851 - val_loss : 1.5505 - val_acc: 0.4383
…
Epoch : 332 - loss : 1.1774 - acc: 0.4914 - val_loss : 1.5670 - val_acc: 0.4401
…
Epoch : 377 - loss : 1.1443 - acc: 0.5001 - val_loss : 1.6314 - val_acc: 0.4208
Epoch : 378 - loss : 1.1453 - acc: 0.4984 - val_loss : 1.5440 - val_acc: 0.4403
…
Epoch : 403 - loss : 1.1344 - acc: 0.5002 - val_loss : 1.6524 - val_acc: 0.4302
Epoch : 404 - loss : 1.1341 - acc: 0.5021 - val_loss : 1.6363 - val_acc: 0.4299
Epoch : 405 - loss : 1.1312 - acc: 0.5026 - val_loss : 1.5418 - val_acc: 0.4461
…
Epoch : 470 - loss : 1.1194 - acc: 0.5026 - val_loss : 1.5144 - val_acc: 0.4542
Epoch : 471 - loss : 1.1160 - acc: 0.5052 - val_loss : 1.5371 - val_acc: 0.4484
…
Epoch : 499 - loss : 1.1105 - acc: 0.5068 - val_loss : 1.5045 - val_acc: 0.4534
Epoch : 500 - loss : 1.1125 - acc: 0.5071 - val_loss : 1.4336 - val_acc: 0.4652
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

In ARC the `val_acc` is `0.4652` after 500 epochs. This is not good.
Ways to improve the `val_acc`.
The size of the ARC data is much smaller than the Cats & Dogs data.
If I generate even more permutations, to get to the same size.
Since ARC has more classifications than the cats & dogs, it may require even more training data to get to a comparable accuracy.

## Future plans

Integrate with HuggingFace's "Trainer".
https://huggingface.co/docs/transformers/main_classes/trainer

More data. 
- Currently only 6 tasks. Add more tasks.
- Permute the existing tasks even more.

Visualize the predicted output.


