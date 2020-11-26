# autoencoder
Input (128x128x3) => Latent (256x1) => Output (128x128x3): compression ratio of  128*128*3/256 = 192. \  
code in train.py\
Trained on 200 epochs, batch size of 256, L2 loss, and 12 layers (including transposed, so half_dim=5).
The input passes through <half_dim/enc_depth> conv layers with stride=2, each shrinks the input by 4. Then, it passes through a fully connected layer (which is implemented as a conv layer with full kernel size) to a latent vector, and from there passes through each of the mentioned layers, but transposed, in reverse order.
![Demo](./demo.png "Reconstruction")
