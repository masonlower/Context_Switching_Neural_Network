# RNN Context Training Project

A neural network implementation for context-dependent decision making using a T-maze task environment. Task based on Akhil Bandi task, publication to come. Mice are given visual and auditory stimuli at the same time. Only one of these are relevant for a block of tasks and this changes once mice reach an accuracy threshold. Code was adapted from [Engel Lab Latent Net](https://github.com/engellab/latent-net) github repository.

## Project Structure

- `UniToMulti.ipynb`/`MultitoUni.ipynb`: Jupyter notebooks for training models on tmaze tasks.
- `Tmaze_UniToMulti.py`: Implements the T-maze task with option for unisensory or multisensory inputs that can be easily switched.
- `lstm_net.py`: Implements LSTM network that will train on Tmaze task. Has option for Dale's Law as well as balance of excitatory and inhibitory ratio.
    - `connectivity_lstm.py`: dependency for lstm_net determines rules units follow when interacting
    - `connectivity.py`: Engel Lab implemntation of connectivity_lstm
- `LSTM_Analysis.py`/`plotting_functions.py`: plotting functions to aid in analyzing networks.  LSTM_analysis useful for testing if network is infering context.  plotting functions useful for determining if network is trained on task.



## Features

- Supports both unimodal and multimodal inputs
- Configurable context switching between visual and audio cues
- Customizable trial parameters and coherence levels
- Generates input streams with controlled noise levels

## Usage

The environment generates trials with the following properties:
- Context switching between visual and audio modalities
- Adjustable coherence levels for both modalities
- Configurable trial timing and decision periods
- Supports both unimodal and multimodal training scenarios

## Dependencies

- NumPy
- PyTorch
- Pandas
- Scipy
- Seaborn

## License

This project is licensed under the MIT License.
