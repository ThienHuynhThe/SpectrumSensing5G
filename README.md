# Intelligent Spectrum Sensing Using CNN to Identify 5G NR and LTE Signals
The evolution towards fifth-generation wireless (5G) and beyond has significantly increased the demand for efficient spectrum management and utilization. Conventional spectrum sensing methods have struggled to accurately characterize spectrum occupancy, particularly when different radio signals share the same frequency band.
To address this challenge, we propose a novel spectrum sensing method by exploiting short-time Fourier transform and neural networks for learning spectrogram patterns. Leveraging encoder-decoder architectures, we design a semantic segmentation network, namely SRNet, to precisely detect multiple signals within a spectrum by identifying spectral content based on the frequency and time occupied by the signals. By incorporating an attention mechanism and multi-scale feature extraction, SRNet effectively learns spectral features and improves segmentation efficiency. Extensive simulations demonstrate SRNet's robustness and effectiveness in identifying 5G New Radio and LTE signals, under challenging channel and radio frequency impairments, making it a promising solution for next-generation spectrum sensing.

<img src="https://github.com/ThienHuynhThe/SpectrumSensing_5GLTE/blob/main/framework.png" height="286px" width="710px" >

The Matlab code and dataset provided here are included in the under-review paper at IEEE Wireless Communications Letters

Thien Huynh-The, Gia-Vuong Nguyen, Thai-Hoc Vu, Daniel Benevides da Costa, and Quoc-Viet Pham, "SRNet: Deep Semantic Segmentation Network for Spectrum Sensing in Wireless Communications," IEEE WCL, 2024.

The dataset can be download on [Google Drive](https://drive.google.com/drive/folders/1DI4dicM65Mix4HgkAnl628AUnUCaZooo?usp=sharing) (please report if not available) 

If there is any error or need to be discussed, please email to [Thien Huynh-The](https://sites.google.com/site/thienhuynhthe/home) via thienht@hcmute.edu.vn
