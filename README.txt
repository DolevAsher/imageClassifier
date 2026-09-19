Gili Raffaeli,Nir Cohen,Roy Shavit,Dolev Asher
326683943,209720838,322558222,211775770

​​Our architecture is built of 5 stages. Each stage consists of 5 steps: convolution -> batch normalization -> ReLU activation -> ResBlock (residual block) -> max pooling.
Finally, an adaptive average pool and a dense classifier map the features to the 20 output classes.

Manipulations on images: horizontal flip, vertical flip, rotation (up to 15 degrees), affine projections, grayscale, color jitter, color inversion, gaussian blur.

We trained a basic model with light manipulations, then saved the weights. Afterwards we fine-tuned the existing model with heavier manipulations while validating that the model still recognizes unmodified images.

Pre-fine-tuning the model recognized 88.4% of unmodified test images, and 63.5% of modified test images. After the fine-tuning the model recognized 89.2% of unmodified test images, and 81.7% of modified test images.