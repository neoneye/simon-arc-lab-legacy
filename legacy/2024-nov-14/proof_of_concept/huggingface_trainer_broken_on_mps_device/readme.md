# HuggingFace's Trainer combined with ViT - Status: Slow CPU, Not working on MPS

Only CPU support. No hardware acceleration on macOS. I need hardware acceleration.

Problem: I can't get `device="mps"` working with HuggingFace's Trainer.
https://github.com/huggingface/transformers/issues/17971
https://stackoverflow.com/questions/76589840/cant-run-transformer-fine-tuning-with-m1-mac-cpu

---

Video describing how the Trainer works with ViT.
https://youtu.be/wao7HRgtcaU?t=874

Code for the video. That I'm basing my code on.
https://colab.research.google.com/github/nateraw/huggingface-hub-examples/blob/main/vit_image_classification_explained.ipynb

However I'm on a Macbook Pro M1, and I cannot run cuda.
Looing at huggingface's TrainerAugments, and it's a huge ugly class.
I'm tempted to not use HuggingFace's Trainer, and instead find an alternative.
