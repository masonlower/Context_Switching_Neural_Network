# RNN Context Training Project

<<<<<<< HEAD
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


=======
A neural network implementation for context-dependent decision making using a T-maze task environment. Task based on Akhil Bandi task, publication to come. Mice are given visual and auditory stimuli at the same time. Only one of these are relevant for a block of tasks and this changes once mice reach an accuracy threshold. Code was adapted from [Engel Lab Latent Net](https://github.com/engellab/latent-net) github repository.

## Project Structure

- `UniToMulti.ipynb`/`MultitoUni.ipynb`: Jupyter notebooks for training models on tmaze tasks.
- `Tmaze_UniToMulti.py`: Implements the T-maze task with option for unisensory or multisensory inputs that can be easily switched.
- `lstm_net.py`: Implements LSTM network that will train on Tmaze task. Has option for Dale's Law as well as balance of excitatory and inhibitory ratio.
    - `connectivity_lstm.py`: dependency for lstm_net determines rules units follow when interacting
    - `connectivity.py`: Engel Lab implemntation of connectivity_lstm
- `LSTM_Analysis.py`/`plotting_functions.py`: plotting functions to aid in analyzing networks.  LSTM_analysis useful for testing if network is infering context.  plotting functions useful for determining if network is trained on task.

- `ContextTraining/`: partial clone of base repo.  Work in progress to allow for context and trial counter to be inputs to network.


## On Going Work

This repository is still a work in progress, activities for some trained models as well as where this is heading before being pushed to github can be followed [here](https://docs.google.com/document/d/12hmnnvVeFA1bg5FMkSGNtCJjN8OK23KWdj3bKicKtl0/edit?tab=t.0)
>>>>>>> 775cd0e61edf08e1fae2a1d773c6ccdae3cf9eff

## Features

- Supports both unimodal and multimodal inputs
- Configurable context switching between visual and audio cues
- Customizable trial parameters and coherence levels
- Generates input streams with controlled noise levels
<<<<<<< HEAD
- Turn context and trial counter inputs on and off (work in progress)
- LSTM temporal dependency, as well as cell and hidden state tracking.
=======
>>>>>>> 775cd0e61edf08e1fae2a1d773c6ccdae3cf9eff

## Usage

The environment generates trials with the following properties:
- Context switching between visual and audio modalities
- Adjustable coherence levels for both modalities
- Configurable trial timing and decision periods
- Supports both unimodal and multimodal training scenarios

## Dependencies

- NumPy
- PyTorch
<<<<<<< HEAD
- Python 3.x
- seaborn
- scipy
- Pandas
=======
- Pandas
- Scipy
- Seaborn
>>>>>>> 775cd0e61edf08e1fae2a1d773c6ccdae3cf9eff

## License

This project is licensed under the MIT License.
