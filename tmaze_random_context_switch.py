

import numpy as np
#%pip install scipy
from scipy.sparse import random
import torch
import copy
from scipy import stats
import scipy.ndimage
import random
from collections import deque



class TmazeEnv:
    
    

    def generate_input_target_stream(context, visual_coh, audio_coh, baseline, alpha, sigma_in, n_t,
                                    stim_on,stim_off, dec_on, dec_off):
        """
        Generate input and target sequence for a given set of trial conditions.

        :param t:
        :param tau:
        :param cue:
        :param visual_coh:
        :param audio_coh:
        :param baseline:
        :param alpha:
        :param sigma_in:
        :param cue_on:
        :param cue_off:
        :param stim_on:
        :param stim_off:
        :param dec_off:
        :param dec_on:

        :return: input stream
        :return: target stream

        """
        # Convert trial events to discrete time

        # Transform coherence to signal
        visual_r = (1 + visual_coh) / 2
        visual_l = 1 - visual_r
        audio_r = (1 + audio_coh) / 2
        audio_l = 1 - audio_r

        #We don't need a cue to signal context, net will have to figure out
        # Cue input stream
        #cue_input = np.zeros([n_t, 6])
        #if context == "visual":
         #   cue_input[cue_on:cue_off, 0] = 1.2 * np.ones(
          #      [cue_off - cue_on, 1]).squeeze()
        #else:
         #   cue_input[cue_on:cue_off, 1] = 1.2 * np.ones(
          #      [cue_off - cue_on, 1]).squeeze()


        # Motion input stream
        visual_input = np.zeros([n_t, 6])
        visual_input[stim_on - 1:stim_off, 2] = visual_r * np.ones([stim_off - stim_on + 1])
        visual_input[stim_on - 1:stim_off, 3] = visual_l * np.ones([stim_off - stim_on + 1])

        # Color input stream
        audio_input = np.zeros([n_t, 6])
        audio_input[stim_on - 1:stim_off, 4] = audio_r * np.ones([stim_off - stim_on + 1])
        audio_input[stim_on - 1:stim_off, 5] = audio_l * np.ones([stim_off - stim_on + 1])

        # Noise and baseline signal
        noise = np.sqrt(2 / alpha * sigma_in * sigma_in) * np.random.multivariate_normal(
            [0, 0, 0, 0, 0, 0], np.eye(6), n_t)
        baseline = baseline * np.ones([n_t, 6])

        # Input stream is rectified sum of baseline, task and noise signals.
        input_stream = np.maximum(baseline + visual_input + audio_input + noise, 0)

        # Target stream
        target_stream = 0.2 * np.ones([n_t, 2])
        if (context == "visual" and visual_coh > 0) or (context == "audio" and audio_coh > 0):
            target_stream[dec_on - 1:dec_off, 0] = 1.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()
            target_stream[dec_on - 1:dec_off, 1] = 0.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()
        else:
            target_stream[dec_on - 1:dec_off, 0] = 0.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()
            target_stream[dec_on - 1:dec_off, 1] = 1.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()

        return input_stream, target_stream


    def generate_trials(n_trials, alpha, sigma_in, baseline, n_coh, n_t=75):
        """
        Create a set of trials consisting of inputs, targets and trial conditions.

        :param n_trials: number of trials per condition.
        :param alpha:
        :param sigma_in:
        :param baseline:
        :param n_coh:
        :param n_t: number of time steps per trial

        :return: inputs, targets, mask, conditions
        """

        cohs = np.linspace(-.2, .2, n_coh)

        stim_on = int(round(n_t * .4))
        stim_off = int(round(n_t))
        dec_on = int(round(n_t * .75))
        dec_off = int(round(n_t))

        inputs = []
        targets = []
        conditions = []

        context = random.choice(["visual", "audio"])
        trials_before_switch = int(np.random.normal(50, 10))
        trial_count = 0

        for visual_coh in cohs:
            for audio_coh in cohs:
                for i in range(n_trials):
                    correct_choice = 1 if ((context == "visual" and visual_coh > 0) or 
                                           (context == "audio" and audio_coh > 0)) else -1
                    conditions.append({"context": context,
                                       "visual_coh": visual_coh,
                                       "audio_coh": audio_coh,
                                       "correct_choice": correct_choice})
                    input_stream, target_stream = TmazeEnv.generate_input_target_stream(context,
                                                                                        visual_coh,
                                                                                        audio_coh,
                                                                                        baseline,
                                                                                        alpha,
                                                                                        sigma_in,
                                                                                        n_t,
                                                                                        stim_on,
                                                                                        stim_off,
                                                                                        dec_on,
                                                                                        dec_off)
                    inputs.append(input_stream)
                    targets.append(target_stream)

                    trial_count += 1
                    if trial_count >= trials_before_switch:
                        context = "audio" if context == "visual" else "visual"
                        trials_before_switch = int(np.random.normal(50, 10))
                        trial_count = 0

        inputs = np.stack(inputs, 0)
        targets = np.stack(targets, 0)

        perm = np.random.permutation(len(inputs))
        inputs = torch.tensor(inputs[perm, :, :], dtype=torch.float32)
        targets = torch.tensor(targets[perm, :, :], dtype=torch.float32)
        conditions = [conditions[index] for index in perm]

        mask = torch.ones_like(targets)
        mask[:, :dec_on, :] = 0

        return inputs, targets, mask, conditions
    
""" Old Generate Trials function, never worked? How Engel had it
for context in ["visual", "audio"]: #need to edit this to incorporate stable context until criteria met
            for visual_coh in cohs:
                #for audio_coh in cohs:
                    for i in range(n_trials):
                        correct_choice = 1 if ((context == "visual" and visual_coh > 0) or (context == "audio" and audio_coh > 0)) else -1
                        input_stream, target_stream = generate_input_target_stream(context,
                                                                                visual_coh,
                                                                                audio_coh,
                                                                               alpha,
                                                                                sigma_in,
                                                                                n_t,
                                                                                cue_on,
                                                                                cue_off,
                                                                                stim_on,
                                                                                stim_off,
                                                                                dec_on,
                                                                                dec_off)
                        inputs.append(input_stream)
                        targets.append(target_stream)
        inputs = np.stack(inputs, 0)
        targets = np.stack(targets, 0)

        perm = np.random.permutation(len(inputs))
        inputs = torch.tensor(inputs[perm, :, :]).float()
        targets = torch.tensor(targets[perm, :, :]).float()
        conditions = [conditions[index] for index in perm]

        # training_mask = np.append(range(stim_on),
        #                           range(dec_on - 1, dec_off - 1))

        mask = torch.ones_like(targets)
        mask[:, :dec_on, :] = 0



        return inputs, targets, mask, conditions
"""

class WeightsThroughSwitches:
    def run_context_transitions(self, net, conditions, total_trials=1000, post_switch_delay=5, trials_per_context=50):
        net.train()

        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
        
        current_context = None
        context_switch_points = []
        weights_snapshots = []
        trials = []
    
        # Initialize hidden state
        batch_size = 1
        hidden = (
            torch.zeros(1, batch_size, net.n).to(net.device),
            torch.zeros(1, batch_size, net.n).to(net.device)
        )
    
        # Create a list of contexts that switches every trials_per_context
        trial_contexts = []
        unique_contexts = list(set(cond["context"] for cond in conditions))
        for i in range(0, total_trials, trials_per_context):
            context_idx = (i // trials_per_context) % len(unique_contexts)
            trial_contexts.extend([unique_contexts[context_idx]] * min(trials_per_context, total_trials - i))
    
        for i in range(total_trials):
            trial_context = trial_contexts[i]
        
            # Find matching condition from original conditions list
            matching_conditions = [c for c in conditions if c["context"] == trial_context]
            current_condition = matching_conditions[i % len(matching_conditions)]
        
            if trial_context != current_context:
                if current_context is not None:
                    weights_snapshots.append({
                        'trial': i-1,
                        'timing': 'pre-switch',
                        'from_context': current_context,
                        'to_context': trial_context,
                        'lstm_ih': copy.deepcopy(net.lstm.weight_ih_l0.data),
                        'lstm_hh': copy.deepcopy(net.lstm.weight_hh_l0.data),
                        'output': copy.deepcopy(net.output_layer.weight.data),
                        'hidden_state': (hidden[0].clone(), hidden[1].clone())
                    })
                    context_switch_points.append(i)
                current_context = trial_context
        
            # Generate input based on trial conditions
            input_stream, target_stream = TmazeEnv.generate_input_target_stream(
                trial_context,
                current_condition["visual_coh"],
                current_condition["audio_coh"],
                baseline=0.2,
                alpha=0.2,
                sigma_in=0.1,
                n_t=75,
                stim_on=30,
                stim_off=75,
                dec_on=56,
                dec_off=75
            )
        
            input_tensor = torch.tensor(input_stream).float().unsqueeze(0)
            target_tensor = torch.tensor(target_stream).float().unsqueeze(0)

            optimizer.zero_grad()
        
            # Forward pass
            output, hidden = net(input_tensor, hidden)
        
            # Compute loss (using the same loss function as in training)
            loss = net.loss_function(output, target_tensor, torch.ones_like(target_tensor))
        
            # Backward pass
            loss.backward()
        
            # Update weights
            optimizer.step()
        
            # Apply connectivity constraints
            net.connectivity_constraints()
        
            # Save post-switch weights after delay
            if i in [x + post_switch_delay for x in context_switch_points]:
                weights_snapshots.append({
                    'trial': i,
                    'timing': 'post-switch',
                    'context': current_context,
                    'lstm_ih': copy.deepcopy(net.lstm.weight_ih_l0.data),
                    'lstm_hh': copy.deepcopy(net.lstm.weight_hh_l0.data),
                    'output': copy.deepcopy(net.output_layer.weight.data),
                    'hidden_state': (hidden[0].clone(), hidden[1].clone())
                })
        
            trials.append({
                'trial': i,
                'context': trial_context,
                'input': input_stream,
                'output': output.squeeze().detach().numpy(),
                'hidden_state': hidden[0].squeeze().detach().numpy()
            })
    
        return trials, context_switch_points, weights_snapshots