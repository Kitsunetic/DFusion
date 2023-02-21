# DFusion

A library to make it easy to implement diffusion models.

It contains:
- `DDPMTrainer`, `DDPMSampler`: (https://arxiv.org/abs/2006.11239)
- `DDIMSampler`: (https://arxiv.org/abs/2010.02502)
- `HeunSampler`: (https://arxiv.org/abs/2206.00364)

Currently working on:
- [x] Include DDPM Trainer and Sampler
- [x] Include DDIM Sampler
- [x] Include Elucidating Diffusion
- [ ] Include D3PM
- [x] Test on CIFAR10 dataset
- [ ] Show CIFAR10 test results


# Installation

```sh
pip install git+https://github.com/Kitsunetic/DFusion.git
```


# Acknowledgement

I borrwed source codes largely from the following repositories.
Many codes are mixtured and done refactoring.
- https://github.com/w86763777/pytorch-ddpm
- https://github.com/lucidrains/denoising-diffusion-pytorch
- https://github.com/google-research/google-research
- https://github.com/openai/point-e
