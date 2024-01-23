# Neural Dissection: interpreting and modifying the behavior of deep audio generative models

This repository accompanies the report for the machine learning project *Neural Dissection: interpreting and modifying the behavior of deep audio generative models*.

## Structure of the repository

There are 3 main folders:
- `biblio`: Articles that served as starting points for our work.
- `code`: Where our coding work is (see next section).
- `report`: Useful material for the report (mostly figures).

The `code` folder is divided into 3 subfolders:
- `models`: Checkpoints are saved here after training (the `backup` folder stores the latest checkpoint during training).
- `notebooks`: Interactive notebooks, mainly for feature visualization. They call scripts from the `scripts` folder.
- `scripts`: function and class definitions for the `GANs` and `visualization`. For the `GANs`, the scripts `dcgan_mnist.py` and `wgan_mnist.py` are used to train the DCGAN and WGAN respectively, or generate a batch of data from noise if the variable `do_train`is set to `False`.

## Setup

Setup a virtual environment with Python 3.10 in the dedicated folder and install requirements:

```bash
python3.10 -m venv ./venv
source `./venv/bin/activate` on Mac
`./venv/Scripts/activate.bat` on Windows 
pip install -r ./code/config/requirements.txt
```

## Paper abstract

In the dynamic landscape of deep learning, neural networks have made remarkable strides, yet their inherent complexity often casts them as enigmatic "black boxes." This investigation marks an advance by extending feature visualization techniques, traditionally employed in image-based models, to GANSynth—a model trained on spectrograms. GANSynth showcases its capability by generating audio waveforms with remarkable precision, effectively capturing and reproducing instantaneous frequency variations in a range of synthesized sounds. Departing from the customary emphasis on visual data, feature visualization not only enhances our understanding of auditory representations but also explores interpretability and diversity within the audio domain. While visualization techniques have been developed, their application has been predominantly confined to the image domain until now. Through the optimization of audio spectrograms and iterative adjustments to inputs, this study reveals specific behaviors within GANSynth, showcasing the efficacy of feature visualization in unraveling the intricacies of neural networks. In summary, this investigation paves the way for a deeper understanding of neural networks trained on non-image data, providing a fundamental tool to demystify their internal workings beyond waveform analysis.


*--- Made with love by Pablo Dumenil, Pharoah Jardin, Aurélien Manté and Benjamin Quiédeville (ATIAM 23-24, Ircam).*