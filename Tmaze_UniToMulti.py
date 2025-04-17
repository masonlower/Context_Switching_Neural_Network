import numpy as np
import torch
import random

class TmazeEnv2:

    
    
    def generate_input_target_stream(context, v_coh, a_coh, baseline, alpha, sigma_in, n_t,
                                   stim_on, stim_off, dec_on, dec_off, modality='multi'):
        """
        Generate input and target sequence for a given set of trial conditions.
        
        Args:
            context (str): 'visual' or 'audio'
            v_coh (float): visual coherence
            a_coh (float): audio coherence
            baseline (float): baseline input level
            alpha (float): scaling factor for noise
            sigma_in (float): standard deviation of input noise
            n_t (int): number of time steps
            modality (str): 'uni' for unimodal or 'multi' for multimodal input
        """
        # Transform coherence to signal
        visual_r = (1 + v_coh) / 2
        visual_l = 1 - visual_r
        audio_r = (1 + a_coh) / 2
        audio_l = 1 - audio_r

        # Initialize input streams
        visual_input = np.zeros([n_t, 4])
        audio_input = np.zeros([n_t, 4])

        # Set inputs based on modality and context
        if modality == 'uni':
            if context == "visual":
                visual_input[stim_on - 1:stim_off, 0] = visual_r * np.ones([stim_off - stim_on + 1])
                visual_input[stim_on - 1:stim_off, 1] = visual_l * np.ones([stim_off - stim_on + 1])

                audio_input[stim_on - 1:stim_off, 2] = 0 * np.ones([stim_off - stim_on + 1])
                audio_input[stim_on - 1:stim_off, 3] = 0 * np.ones([stim_off - stim_on + 1])
                # Audio inputs the value after the '='
            else:  # audio context
                audio_input[stim_on - 1:stim_off, 2] = audio_r * np.ones([stim_off - stim_on + 1])
                audio_input[stim_on - 1:stim_off, 3] = audio_l * np.ones([stim_off - stim_on + 1])

                visual_input[stim_on - 1:stim_off, 0] = 0 * np.ones([stim_off - stim_on + 1])
                visual_input[stim_on - 1:stim_off, 1] = 0 * np.ones([stim_off - stim_on + 1])
                # Visual inputs value after '='
        else:  # multimodal
            visual_input[stim_on - 1:stim_off, 0] = visual_r * np.ones([stim_off - stim_on + 1])
            visual_input[stim_on - 1:stim_off, 1] = visual_l * np.ones([stim_off - stim_on + 1])
            audio_input[stim_on - 1:stim_off, 2] = audio_r * np.ones([stim_off - stim_on + 1])
            audio_input[stim_on - 1:stim_off, 3] = audio_l * np.ones([stim_off - stim_on + 1])

        # Add noise and baseline
        noise = np.sqrt(2 / alpha * sigma_in * sigma_in) * np.random.multivariate_normal(
            [0, 0, 0, 0], np.eye(4), n_t)
        baseline = baseline * np.ones([n_t, 4])

        # Combine inputs
        input_stream = np.maximum(baseline + visual_input + audio_input + noise, 0)

        # Target stream (same for both modalities)
        target_stream = 0.2 * np.ones([n_t, 2])
        if (context == "visual" and v_coh > 0) or (context == "audio" and a_coh > 0):
            target_stream[dec_on - 1:dec_off, 0] = 1.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()
            target_stream[dec_on - 1:dec_off, 1] = 0.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()
        else:
            target_stream[dec_on - 1:dec_off, 0] = 0.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()
            target_stream[dec_on - 1:dec_off, 1] = 1.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()

        return input_stream, target_stream

    def generate_trials(n_trials, alpha, sigma_in, baseline, n_coh, modality='multi', n_t=75, trials_per_context=50):
        """
        Create a set of trials with context switching.
        """
        cohs = np.linspace(-1, 1, n_coh)
    
        stim_on = int(round(n_t * .4))
        stim_off = int(round(n_t))
        dec_on = int(round(n_t * .75))
        dec_off = int(round(n_t))
    
        inputs = []
        targets = []
        conditions = []
    
        context = random.choice(["visual", "audio"])
        trial_count = 0
        total_generated = 0
        block_number = 0

        all_coherences = [(v_coh, a_coh) for v_coh in cohs for a_coh in cohs]
        
     # Generate trials 
        while total_generated < n_trials:
            v_coh, a_coh = all_coherences[total_generated % len(all_coherences)]  # Cycle through coherences
                # Switch context if needed
            if trial_count >= trials_per_context:
                context = "audio" if context == "visual" else "visual"
                trial_count = 0
                block_number += 1
                
            v_coh = random.choice(cohs)
            a_coh = random.choice(cohs)
            if modality == 'uni':
                if context == "visual":
                    visual_r = (1 + v_coh) / 2
                    visual_l = 1 - visual_r
                    audio_r = 0
                    audio_l = 0
                    correct_choice = 1 if visual_r > visual_l else -1
                else: #audio
                    audio_r = (1 + a_coh) / 2
                    audio_l = 1 - audio_r
                    visual_r = 0
                    visual_l = 0
                    correct_choice = 1 if audio_r > audio_l else -1
            else: #multi
                visual_r = (1 + v_coh) / 2
                visual_l = 1 - visual_r
                audio_r = (1 + a_coh) / 2
                audio_l = 1 - audio_r
                if context == "visual":
                    correct_choice = 1 if visual_r > visual_l else -1
                else:  # audio
                    correct_choice = 1 if audio_r > audio_l else -1
        
            input_stream, target_stream = TmazeEnv2.generate_input_target_stream(
                context, v_coh, a_coh, baseline, alpha, sigma_in,
                n_t, stim_on, stim_off, dec_on, dec_off, modality
            )
        
            inputs.append(input_stream)
            targets.append(target_stream)

            conditions.append({
                "context":context,
                "v_coh":v_coh,
                "a_coh":a_coh,
                "visual_r":visual_r,
                "visual_l":visual_l,
                "audio_r":audio_r,
                "audio_l":audio_l,
                "modality":modality,
                "correct_choice": correct_choice,
                "block_number": block_number
            })
                
            total_generated += 1
            trial_count += 1
                    

        # Convert to tensors and apply permutation
        inputs = np.stack(inputs, 0)
        targets = np.stack(targets, 0)

        #perm = np.random.permutation(len(inputs))
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        #conditions = [conditions[index] for index in perm]

        mask = torch.ones_like(targets)
        mask[:, :dec_on, :] = 0

        return inputs, targets, mask, conditions