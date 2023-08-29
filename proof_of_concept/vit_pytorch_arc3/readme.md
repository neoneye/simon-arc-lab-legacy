# Proof of concept - Vision transformer with ARC tasks, and predict a large area, and visualization - Status: Not working

Onehot encoding of the input image.

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

This is with 6 ARC tasks: 10 epochs takes 30m on a M1 Mac. 

Each task have 10.000 images in the `train` dir. And there are 10 classes: color0..color9.
The `tasks.zip` is 84mb. The images are highly similar.

```
Epoch : 1 - loss : 2.0155 - acc: 0.2973 - val_loss : 1.6470 - val_acc: 0.4620
Epoch : 2 - loss : 1.7632 - acc: 0.4035 - val_loss : 1.5655 - val_acc: 0.4853
Epoch : 3 - loss : 1.7222 - acc: 0.4110 - val_loss : 1.5763 - val_acc: 0.4748
…
Epoch : 8 - loss : 1.5679 - acc: 0.4340 - val_loss : 1.4290 - val_acc: 0.4913
Epoch : 9 - loss : 1.5440 - acc: 0.4356 - val_loss : 1.3713 - val_acc: 0.5057
Epoch : 10 - loss : 1.5261 - acc: 0.4390 - val_loss : 1.3692 - val_acc: 0.5031
…
Epoch : 18 - loss : 1.4177 - acc: 0.4548 - val_loss : 1.2735 - val_acc: 0.5061
Epoch : 19 - loss : 1.4084 - acc: 0.4565 - val_loss : 1.2667 - val_acc: 0.4979
Epoch : 20 - loss : 1.3979 - acc: 0.4573 - val_loss : 1.2535 - val_acc: 0.5124
…
Epoch : 38 - loss : 1.3308 - acc: 0.4697 - val_loss : 1.2322 - val_acc: 0.5072
Epoch : 39 - loss : 1.3202 - acc: 0.4695 - val_loss : 1.2222 - val_acc: 0.5095
Epoch : 40 - loss : 1.3177 - acc: 0.4722 - val_loss : 1.2143 - val_acc: 0.5087
…
Epoch : 58 - loss : 1.2774 - acc: 0.4795 - val_loss : 1.2011 - val_acc: 0.5075
Epoch : 59 - loss : 1.2743 - acc: 0.4802 - val_loss : 1.2063 - val_acc: 0.5033
Epoch : 60 - loss : 1.2686 - acc: 0.4825 - val_loss : 1.1869 - val_acc: 0.5183
…
Epoch : 78 - loss : 1.2500 - acc: 0.4833 - val_loss : 1.1932 - val_acc: 0.5050
Epoch : 79 - loss : 1.2401 - acc: 0.4846 - val_loss : 1.1923 - val_acc: 0.5012
Epoch : 80 - loss : 1.2422 - acc: 0.4841 - val_loss : 1.1818 - val_acc: 0.5130
```

Training with another dataset:
```
Epoch : 1 - loss : 1.5849 - acc: 0.4272 - val_loss : 1.3754 - val_acc: 0.4602
Epoch : 2 - loss : 1.3872 - acc: 0.4648 - val_loss : 1.2827 - val_acc: 0.4909
Epoch : 3 - loss : 1.3301 - acc: 0.4735 - val_loss : 1.2405 - val_acc: 0.4934
…
Epoch : 58 - loss : 1.0924 - acc: 0.5164 - val_loss : 1.0603 - val_acc: 0.5260
Epoch : 59 - loss : 1.0963 - acc: 0.5115 - val_loss : 1.0611 - val_acc: 0.5202
Epoch : 60 - loss : 1.0936 - acc: 0.5153 - val_loss : 1.0762 - val_acc: 0.5207
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

In ARC the `val_acc` is `0.5031` after 10 epochs. This is not good.
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


