# RNN Context Training Project

A neural network implementation for context-dependent decision making using a T-maze task environment. Task based on Akhil Bandi task, publication to come. Mice are given visual and auditory stimuli at the same time. Only one of these are relevant for a block of tasks and this changes once mice reach an accuracy threshold. Code was originally adapted from [Engel Lab Latent Net](https://github.com/engellab/latent-net) github repository.

## Project Structure

- `MultitoUni.ipynb / UniToMulti.ipynb`: Jupyter notebooks for training and running basic analysis on LSTM NN, can easily be modified to become one another only two seperate files for size management. Can be used as tutorial for familiarization with the project.

- `Tmaze_UniToMulti.py`: Implements the T-maze task with option for unisensory or multisensory inputs
    - `tmaze_random_context_switch.py`: Tmaze environment, but context switches are random around an average trial length, should probably be reserved for when network starts infering context on above implementation.
- `lstm_net.py`: Implements LSTM network that will train on Tmaze task. Has option for Dale's Law as well as balance of excitatory and inhibitory ratio.
    - `connectivity_lstm.py`: implements how units connect with each other, dependency for lstm_net
    - `connectivity.py`: base of connectivity_lstm, not in use but working if revert net back to Engel Lab implementation.
- `plotting_functions.py`: functios to aid in visualizing network performance.
- `LSTM_Analysis`: plotting functions 2.0, functions used for testing whether LSTM is infering context or not

- `ContextTraining/`: partial clone of repository, Tmaze implementation will allow for context and trial counter inputs to be added to NN inputs, as well as to be turned on and off (work in progress).
-`TrainedModels`: Models will tweaked parameters, uni/multi indicates whether model was initially trained on uni or multisensory.  85_15 indicates 85% excitatory units, 15% inhib.  0/2 indicates whether irrelevant stimuli were 0 inputs or .2
-`WeightsAndActivity`: images of above networks weights and output unit activities by trial type. Code for plots are in UniToMulti.ipynb (should have added to plotting functions).



## Features

- Supports both unimodal and multimodal inputs
- Configurable context switching between visual and audio cues
- Customizable trial parameters and coherence levels
- Generates input streams with controlled noise levels
- Turn context and trial counter inputs on and off (work in progress)
- LSTM temporal dependency, as well as cell and hidden state tracking.

## Usage

The environment generates trials with the following properties:
- Context switching between visual and audio modalities
- Adjustable coherence levels for both modalities
- Configurable trial timing and decision periods
- Supports both unimodal and multimodal training scenarios

## Dependencies

- NumPy
- PyTorch
- Python 3.x
- seaborn
- scipy
- Pandas

## License

This project is licensed under the MIT License.
