# Conditional GAN for CIFAR-10 Dataset

This project implements a Conditional Generative Adversarial Network (Conditional GAN) to generate images from the CIFAR-10 dataset. Conditional GANs enhance traditional GANs by conditioning the model on additional information, leading to more specific and controlled image generation.

## Overview

Generative Adversarial Networks (GANs) comprise a generator and a discriminator that engage in a minimax game where the generator attempts to create realistic images, and the discriminator tries to distinguish between real and generated images. In this Conditional GAN, both the generator and discriminator are conditioned on class labels to produce images specific to those labels.

## Methodology

- **Generator**: Uses a series of transposed convolutional layers to generate images.
- **Discriminator**: Uses convolutional layers to classify images as real or fake.
- **Training**: The network is trained through alternating updates to the discriminator using real and generated images and the generator using the feedback from the discriminator.

The training involves leveraging convolutional neural networks (CNNs), batch normalization, leaky ReLU activations, and dropout regularization.

## Usage

1. Clone this repository.
2. Install dependencies.
3. Run the training script:
4. Adjust the number of epochs as needed.

## Results

Due to limited training time and the need for further hyperparameter tuning, the initial results did not show a significant improvement in the quality of generated images. Further training and adjustments are required to enhance the model's performance.

## Limitations

The current model was trained for a limited number of epochs, resulting in lower quality images. Future improvements could involve more extensive training and resource allocation to achieve better results.

## References

- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1411.1784.
- Liu, S., et al. (2020). Diverse image generation via self-conditioned gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

## License

This project is open-sourced under the MIT License.

## Contact

For any queries or discussions regarding improvements, please open an issue on this repository.
