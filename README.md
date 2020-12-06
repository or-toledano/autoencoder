# autoencoder
Input (128x128x3) => Latent (256x1) => Output (128x128x3): reducing the number of entries by 128x128x3/256 = 192 (the latent is float though).\
code in train.py\
Trained on 200 epochs, batch size of 256, L2 loss, Adam optimizer, 12 layers (including transposed, so half_dim=5 conv layers + leaky relu + batch norm for each size, in addition to one FC in each side), on 60k out of 200k of the CelebA dataset.
The input passes through <half_dim/enc_depth> conv layers with stride=2, each shrinks the input by 4. Then, it passes through a fully connected layer (which is implemented as a conv layer with full kernel size) to a latent vector, and from there passes through each of the mentioned layers, but transposed, in reverse order.\
A result from the test set:
![Demo](./demo.png "Reconstruction")
