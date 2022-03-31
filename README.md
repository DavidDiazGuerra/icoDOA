# Direction of Arrival Estimation of Sound Sources Using Icosahedral CNNs

Code repository for the paper _Direction of Arrival Estimation of Sound Sources Using Icosahedral CNNs_
[[1]](#references).

If you're only looking for our Pytorch implementation of the Icosahedral CNNs, you can find it [here](https://github.com/DavidDiazGuerra/icoCNN).

- [Dependencies](#dependencies)
- [Datasets](#datasets)
- [Other source files](#other-source-files)
- [Pretrained models](#pretrained-models)
- [References](#references)

## Dependencies

* Python: it has been tested with Python 3.8.1
* Numpy, matplotlib, scipy, soundfile, pandas and tqdm
* Pytorch: it has been tested with Python 1.8.1
* [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) [[2]](#references)
* [icoCNN](https://github.com/DavidDiazGuerra/icoCNN)
* [webrtcvad](https://github.com/wiseman/py-webrtcvad)

## Datasets

* **LibriSpeech** The training dataset is generated during the training of the models as the trajectories are needed, 
but to simulate them you will need to have the [LibriSpeech corpus](http://www.openslr.org/12) in your machine. By 
default, the main scripts look for it in [datasets/LibriSpeech](https://github.com/DavidDiazGuerra/icoDOA/tree/master/datasets/LibriSpeech) 
but you can modify its phat with the `path_train` and `path_test` variables.
* **LOCATA** In order to test the models with actual recordings, you will also need the dataset of the 
[LOCATA challenge](https://www.locata.lms.tf.fau.de/). By default, the main scripts look for it in 
[datasets/LOCATA](https://github.com/DavidDiazGuerra/icoDOA/tree/master/datasets/LOCATA) 
but you can modify its phat with the `path_locata` variable.

## Main script

You can use the script [1sourceTracking_icoCNN.py](https://github.com/DavidDiazGuerra/icoDOA/blob/master/1sourceTracking_icoCNN.py) 
to train the model and test it with synthetic and real recordings. You can change the resolution of the inputs maps by 
changing the value or `r` in [line 22](https://github.com/DavidDiazGuerra/icoDOA/blob/master/1sourceTracking_icoCNN.py#L22).
The script is organized in cells, you can skip the [training cell](https://github.com/DavidDiazGuerra/icoDOA/blob/master/1sourceTracking_icoCNN.py#L115) 
and just [load the pretrained models](https://github.com/DavidDiazGuerra/icoDOA/blob/master/1sourceTracking_icoCNN.py#L115).

You can find the definition of the model in [acousticTrackingModels.py](https://github.com/DavidDiazGuerra/icoDOA/blob/master/acousticTrackingModels.py#L19)
and the implementation of our sof-argmax function in [acousticTrackingModules.py](https://github.com/DavidDiazGuerra/icoDOA/blob/master/acousticTrackingModules.py#L57).
If you are looking for the implementation of the icosahedral convolutions, they have their own [repository](https://github.com/DavidDiazGuerra/icoCNN).
The baseline model Cross3D [[3]](#references) also has his own [repository](https://github.com/DavidDiazGuerra/Cross3D)
with his code and the pretrained models.

## Pretrained models

The pretrained models and the test results can be found in the [models](https://github.com/DavidDiazGuerra/icoDOA/tree/master/models)
and [results](https://github.com/DavidDiazGuerra/icoDOA/tree/master/results) folders.

## Other source files

`acousticTrackingDataset.py`, `acousticTrackingLearners.py`, `acousticTrackingModels.py` and `acousticTrackingDataset.py`
contain several classes and functions employed by the main script. They are updated versions of the onew found in the 
repository of [Cross3D](https://github.com/DavidDiazGuerra/Cross3D) and have been published to facilitate the replicability 
of the research presented in [[1]](#references), not as a software library. Therefore, any feature included in them that 
is not used by the main script may be untested and could contain bugs.

## References

>[1] D. Diaz-Guerra, A. Miguel, J.R. Beltran, "Direction of Arrival Estimation of Sound Sources Using Icosahedral CNNs," [[arXiv preprint]()].
>
>[2] D. Diaz-Guerra, A. Miguel, J.R. Beltran, "gpuRIR: A python library for Room Impulse Response simulation with GPU 
acceleration," in Multimedia Tools and Applications, Oct. 2020 [[DOI](https://doi.org/10.1007/s11042-020-09905-3)] [[SharedIt](https://rdcu.be/b8gzW)] [[arXiv preprint](https://arxiv.org/abs/1810.11359)].
>
>[3] D. Diaz-Guerra, A. Miguel and J. R. Beltran, "Robust Sound Source Tracking Using SRP-PHAT and 3D Convolutional Neural Networks," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 300-311, 2021 [[DOI](https://doi.org/10.1109/TASLP.2020.3040031)] [[arXiv preprint](https://arxiv.org/abs/2006.09006)].
