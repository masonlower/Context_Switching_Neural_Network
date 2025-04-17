

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import scipy as sy
from net import *
from matplotlib.gridspec import GridSpec

plt.rcParams["axes.grid"] = False


def pf(x, alpha, beta):
    return 1. / (1 + np.exp(-(x - alpha) / beta))


def prob_right(x):
    return np.sum(x > 0) / len(x)



def psychometric(net,u,conditions):
    par0 = np.array([0., 1.])
    contrasts = np.linspace(-1, 1, 15)
    #x = net.forward(u)
    #output = net.output_layer(x)
    output, _ = net.forward(u) 

    rows = []
    for trial in range(u.shape[0]):
        rows.append({'context': conditions[trial]['context'],   
                    'visual_coh': conditions[trial]['v_coh'],
                     'audio_coh': conditions[trial]['a_coh'],
                     'choice': torch.relu(output[trial, -1, 0] - output[trial, -1, 1])})
    df = pd.DataFrame(rows)

    visual_df = df.groupby(['context','visual_coh'])['choice'].apply(prob_right).reset_index(name='prob_right')
    audio_df = df.groupby(['context','audio_coh'])['choice'].apply(prob_right).reset_index(name='prob_right')



    fig = plt.figure(figsize=(4, 1.5))
    gs = gridspec.GridSpec(1, 2,wspace=.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    par, mcov = sy.optimize.curve_fit(pf, visual_df[(visual_df.context=="visual")].visual_coh.values, visual_df[(visual_df.context=="visual")].prob_right.values, par0)
    ax0.plot( 100 * contrasts,100 * pf(contrasts, par[0], par[1]),color='black', lw=1,marker ='.',label = 'visual')

    par, mcov = sy.optimize.curve_fit(pf, visual_df[(visual_df.context == "audio")].visual_coh.values,
                                      visual_df[(visual_df.context == "audio")].prob_right.values, par0)
    ax0.plot(100 * contrasts,100 * pf(contrasts, par[0], par[1]), color='lightgray', lw=1,marker = '.',label = 'Audio')

    par, mcov = sy.optimize.curve_fit(pf, audio_df[(audio_df.context=="visual")].audio_coh.values, audio_df[(audio_df.context=="visual")].prob_right.values, par0)
    ax1.plot( 100 * contrasts,100 * pf(contrasts, par[0], par[1]),color='black', lw=1,marker ='.',label = 'visual')

    par, mcov = sy.optimize.curve_fit(pf, audio_df[(audio_df.context == "audio")].audio_coh.values,
                                      audio_df[(audio_df.context == "audio")].prob_right.values, par0)
    ax1.plot(100 * contrasts,100 * pf(contrasts, par[0], par[1]), color='lightgray', lw=1,marker = '.',label = 'Audio')


    ax1.legend()
    for ax in [ax0,ax1]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_tick_params(labelsize=7, bottom=True)
        ax.yaxis.set_tick_params(labelsize=7, left=True)
    ax0.set_ylabel("Choice to right (%)", fontsize=8)
    ax0.set_xlabel("visual coherence (%)", fontsize=8)
    ax1.set_xlabel("Audio coherence (%)", fontsize=8)

def animate_context_weights(weights_snapshots, context_type='visual', interval=1000):
    """
    Create an animation of network weights before context switches.
    
    Args:
        weights_snapshots (list): List of dictionaries containing weight snapshots
        context_type (str): 'visual' or 'audio'
        interval (int): Time between frames in milliseconds
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    # Filter snapshots for pre-switch and specific context
    relevant_snapshots = [
        snap for snap in weights_snapshots 
        if snap['timing'] == 'pre-switch' and snap['from_context'] == context_type
    ]
    
    if not relevant_snapshots:
        print(f"No pre-switch snapshots found for context: {context_type}")
        return
    
    # Create figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Network Weights Evolution - {context_type} context')
    
    # Initialize plots
    images = []
    for snap in relevant_snapshots:
        img_ih = axes[0].imshow(snap['lstm_ih'].detach().numpy(), cmap='coolwarm', aspect='auto')
        img_hh = axes[1].imshow(snap['lstm_hh'].detach().numpy(), cmap='coolwarm', aspect='auto')
        img_out = axes[2].imshow(snap['output'].detach().numpy(), cmap='coolwarm', aspect='auto')
        images.append([img_ih, img_hh, img_out])
    
    # Set titles
    axes[0].set_title('Input-Hidden Weights')
    axes[1].set_title('Hidden-Hidden Weights')
    axes[2].set_title('Output Weights')
    
    # Add trial number text
    trial_text = fig.text(0.02, 0.95, '', fontsize=10)
    
    def update(frame):
        snap = relevant_snapshots[frame]
        images[frame][0].set_array(snap['lstm_ih'].detach().numpy())
        images[frame][1].set_array(snap['lstm_hh'].detach().numpy())
        images[frame][2].set_array(snap['output'].detach().numpy())
        trial_text.set_text(f'Trial: {snap["trial"]}')
        return images[frame] + [trial_text]
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(relevant_snapshots),
        interval=interval, blit=True
    )
    
    plt.tight_layout()
    return ani

def analyze_weight_similarity(weights_snapshots):
    """
    Analyze similarity between weight matrices before context switches.
    Returns separate analyses for visual and audio contexts.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Filter pre-switch snapshots by context
    visual_snapshots = [snap for snap in weights_snapshots 
                       if snap['timing'] == 'pre-switch' and snap['from_context'] == 'visual']
    audio_snapshots = [snap for snap in weights_snapshots 
                      if snap['timing'] == 'pre-switch' and snap['from_context'] == 'audio']
    
    def compute_similarities(snapshots):
        n_snapshots = len(snapshots)
        # Initialize similarity matrices
        ih_sim = np.zeros((n_snapshots, n_snapshots))
        hh_sim = np.zeros((n_snapshots, n_snapshots))
        out_sim = np.zeros((n_snapshots, n_snapshots))
        
        for i in range(n_snapshots):
            for j in range(n_snapshots):
                # Flatten weights for cosine similarity
                ih1 = snapshots[i]['lstm_ih'].detach().numpy().flatten()
                ih2 = snapshots[j]['lstm_ih'].detach().numpy().flatten()
                hh1 = snapshots[i]['lstm_hh'].detach().numpy().flatten()
                hh2 = snapshots[j]['lstm_hh'].detach().numpy().flatten()
                out1 = snapshots[i]['output'].detach().numpy().flatten()
                out2 = snapshots[j]['output'].detach().numpy().flatten()
                
                ih_sim[i,j] = cosine_similarity(ih1.reshape(1,-1), ih2.reshape(1,-1))
                hh_sim[i,j] = cosine_similarity(hh1.reshape(1,-1), hh2.reshape(1,-1))
                out_sim[i,j] = cosine_similarity(out1.reshape(1,-1), out2.reshape(1,-1))
        
        return ih_sim, hh_sim, out_sim
    
    # Compute similarities for each context
    visual_ih_sim, visual_hh_sim, visual_out_sim = compute_similarities(visual_snapshots)
    audio_ih_sim, audio_hh_sim, audio_out_sim = compute_similarities(audio_snapshots)
    
    # Plot similarities
    def plot_similarities(ih_sim, hh_sim, out_sim, context, trial_numbers):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Weight Similarity Matrices - {context} Context', fontsize=16)
        
        sns.heatmap(ih_sim, ax=axes[0], cmap='coolwarm', center=0)
        axes[0].set_title('Input-Hidden Weight Similarity')
        axes[0].set_xlabel('Switch Number')
        axes[0].set_ylabel('Switch Number')
        
        sns.heatmap(hh_sim, ax=axes[1], cmap='coolwarm', center=0)
        axes[1].set_title('Hidden-Hidden Weight Similarity')
        axes[1].set_xlabel('Switch Number')
        
        sns.heatmap(out_sim, ax=axes[2], cmap='coolwarm', center=0)
        axes[2].set_title('Output Weight Similarity')
        axes[2].set_xlabel('Switch Number')
        
        plt.tight_layout()
        return fig
    
    # Plot for both contexts
    visual_trials = [snap['trial'] for snap in visual_snapshots]
    audio_trials = [snap['trial'] for snap in audio_snapshots]
    
    visual_fig = plot_similarities(visual_ih_sim, visual_hh_sim, visual_out_sim, 
                                 'Visual', visual_trials)
    audio_fig = plot_similarities(audio_ih_sim, audio_hh_sim, audio_out_sim, 
                                'Audio', audio_trials)
    
    # Return similarity matrices and trial numbers for further analysis
    return {
        'visual': {
            'ih_sim': visual_ih_sim,
            'hh_sim': visual_hh_sim,
            'out_sim': visual_out_sim,
            'trials': visual_trials
        },
        'audio': {
            'ih_sim': audio_ih_sim,
            'hh_sim': audio_hh_sim,
            'out_sim': audio_out_sim,
            'trials': audio_trials
        }
    }

def analyze_switch_transitions(weights_snapshots):
    """
    Analyze weight similarities between pre and post switch states.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics.pairwise import cosine_similarity

    # Group snapshots into pre-post pairs
    switch_pairs = []
    for i in range(len(weights_snapshots)-1):
        if (weights_snapshots[i]['timing'] == 'pre-switch' and 
            weights_snapshots[i+1]['timing'] == 'post-switch'):
            switch_pairs.append({
                'from_context': weights_snapshots[i]['from_context'],
                'to_context': weights_snapshots[i]['to_context'],
                'pre': weights_snapshots[i],
                'post': weights_snapshots[i+1],
                'trial': weights_snapshots[i]['trial']
            })

    # Calculate similarities for each layer and each transition
    transitions = []
    for pair in switch_pairs:
        # Calculate cosine similarity directly
        ih_pre = pair['pre']['lstm_ih'].detach().numpy().reshape(1, -1)
        ih_post = pair['post']['lstm_ih'].detach().numpy().reshape(1, -1)
        ih_sim = cosine_similarity(ih_pre, ih_post)[0][0]
        
        hh_pre = pair['pre']['lstm_hh'].detach().numpy().reshape(1, -1)
        hh_post = pair['post']['lstm_hh'].detach().numpy().reshape(1, -1)
        hh_sim = cosine_similarity(hh_pre, hh_post)[0][0]
        
        out_pre = pair['pre']['output'].detach().numpy().reshape(1, -1)
        out_post = pair['post']['output'].detach().numpy().reshape(1, -1)
        out_sim = cosine_similarity(out_pre, out_post)[0][0]

        transitions.append({
            'from_context': pair['from_context'],
            'to_context': pair['to_context'],
            'trial': pair['trial'],
            'ih_sim': ih_sim,
            'hh_sim': hh_sim,
            'out_sim': out_sim
        })

        # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Weight Changes During Context Switches', fontsize=14)

    # Separate visual->audio and audio->visual transitions
    visual_to_audio = [t for t in transitions if t['from_context'] == 'visual']
    audio_to_visual = [t for t in transitions if t['from_context'] == 'audio']

    # Plot for visual->audio transitions
    if visual_to_audio:
        trials_v2a = [t['trial'] for t in visual_to_audio]
        axes[0].plot(trials_v2a, [t['ih_sim'] for t in visual_to_audio], 'b-', label='Input-Hidden')
        axes[0].plot(trials_v2a, [t['hh_sim'] for t in visual_to_audio], 'r-', label='Hidden-Hidden')
        axes[0].plot(trials_v2a, [t['out_sim'] for t in visual_to_audio], 'g-', label='Output')
        axes[0].set_title('Visual → Audio Transitions')
        axes[0].set_xlabel('Trial Number')
        axes[0].set_ylabel('Weight Similarity')
        axes[0].legend()

    # Plot for audio->visual transitions
    if audio_to_visual:
        trials_a2v = [t['trial'] for t in audio_to_visual]
        axes[1].plot(trials_a2v, [t['ih_sim'] for t in audio_to_visual], 'b-', label='Input-Hidden')
        axes[1].plot(trials_a2v, [t['hh_sim'] for t in audio_to_visual], 'r-', label='Hidden-Hidden')
        axes[1].plot(trials_a2v, [t['out_sim'] for t in audio_to_visual], 'g-', label='Output')
        axes[1].set_title('Audio → Visual Transitions')
        axes[1].set_xlabel('Trial Number')
        axes[1].legend()

    plt.tight_layout()
    
    return transitions, fig



def plot_network_activity_analysis(conditions, z):
    """
    Create three plots analyzing network activity:
    1. Average activity by context
    2. Average activity by correct choice
    3. Activity split by both context and correct choice
    
    Args:
        conditions (list): List of dictionaries containing trial conditions
        z (torch.Tensor): Network output tensor (trials x timesteps x output_units)
    """
    z_np = z.detach().numpy()
    time_points = np.arange(z_np.shape[1])
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    
    # Plot 1: Context only
    ax1 = fig.add_subplot(gs[0, 0])
    visual_trials = [z_np[i] for i, cond in enumerate(conditions) if cond['context'] == 'visual']
    audio_trials = [z_np[i] for i, cond in enumerate(conditions) if cond['context'] == 'audio']
    
    visual_mean = np.mean(visual_trials, axis=0)
    audio_mean = np.mean(audio_trials, axis=0)
    
    for unit in range(z_np.shape[2]):
        ax1.plot(time_points, visual_mean[:, unit], 
                 label=f'Visual - Unit {unit}', linestyle='-')
        ax1.plot(time_points, audio_mean[:, unit], 
                 label=f'Audio - Unit {unit}', linestyle='--')
    ax1.set_title('Average Activity by Context')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Activity')
    ax1.legend()
    
    # Plot 2: Correct choice only
    ax2 = fig.add_subplot(gs[0, 1])
    right_trials = [z_np[i] for i, cond in enumerate(conditions) if cond['correct_choice'] == 1]
    left_trials = [z_np[i] for i, cond in enumerate(conditions) if cond['correct_choice'] == -1]
    
    right_mean = np.mean(right_trials, axis=0)
    left_mean = np.mean(left_trials, axis=0)
    
    for unit in range(z_np.shape[2]):
        ax2.plot(time_points, right_mean[:, unit], 
                 label=f'Right - Unit {unit}', linestyle='-')
        ax2.plot(time_points, left_mean[:, unit], 
                 label=f'Left - Unit {unit}', linestyle='--')
    ax2.set_title('Average Activity by Correct Choice')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Activity')
    ax2.legend()
    
    # Plot 3 & 4: Split by context and correct choice
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Visual context split by choice
    visual_right = [z_np[i] for i, cond in enumerate(conditions) 
                   if cond['context'] == 'visual' and cond['correct_choice'] == 1]
    visual_left = [z_np[i] for i, cond in enumerate(conditions) 
                  if cond['context'] == 'visual' and cond['correct_choice'] == -1]
    
    # Audio context split by choice
    audio_right = [z_np[i] for i, cond in enumerate(conditions) 
                  if cond['context'] == 'audio' and cond['correct_choice'] == 1]
    audio_left = [z_np[i] for i, cond in enumerate(conditions) 
                 if cond['context'] == 'audio' and cond['correct_choice'] == -1]
    
    # Calculate means
    vr_mean = np.mean(visual_right, axis=0)
    vl_mean = np.mean(visual_left, axis=0)
    ar_mean = np.mean(audio_right, axis=0)
    al_mean = np.mean(audio_left, axis=0)
    
    # Plot visual context split by choice
    for unit in range(z_np.shape[2]):
        ax3.plot(time_points, vr_mean[:, unit], 
                 label=f'Right - Unit {unit}', linestyle='-')
        ax3.plot(time_points, vl_mean[:, unit], 
                 label=f'Left - Unit {unit}', linestyle='--')
    ax3.set_title('Visual Context by Choice')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Activity')
    ax3.legend()
    
    # Plot audio context split by choice
    for unit in range(z_np.shape[2]):
        ax4.plot(time_points, ar_mean[:, unit], 
                 label=f'Right - Unit {unit}', linestyle='-')
        ax4.plot(time_points, al_mean[:, unit], 
                 label=f'Left - Unit {unit}', linestyle='--')
    ax4.set_title('Audio Context by Choice')
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Activity')
    ax4.legend()
    
    plt.tight_layout()
    return fig



def moving_window (net, trials = 1000, window_size = 40, 
                   trials_per_context = 100, plotting = True):
    """
    Track and plot model performance through a series of context switches 
    in training mode

    Args:
        net: the model
        trials: total number of trials:
        window_size: the number of trials used to calculate accuracy (last X trials)
        trials_per_context: number of trials per context before a switch

    """

    
    import torch.nn as nn
    import Tmaze_UniToMulti as TMUM
    import matplotlib.pyplot as plt
    import seaborn as sns
    import importlib
    from collections import deque


    net.train()

    u_train, z_train, mask_train, conditions_window = TMUM.TmazeEnv2.generate_trials(trials, alpha=0.5, sigma_in=0.1, baseline=0.2, n_coh=8, 
                                                                  modality='multi', n_t=75, trials_per_context=trials_per_context)
    
    device = next(net.parameters()).device
    u_train = u_train.to(device)

    accuracies = []
    recent_correct = deque(maxlen=window_size)
    context_switches = []

    outputs, _ = net(u_train)
    decisions = outputs[:, -1, :]
    predictions = (decisions[:, 0] > decisions[:, 1]).float()
    correct_choices = torch.tensor([1 if c['correct_choice'] == 1 else 0 
                                  for c in conditions_window], 
                                  device=device)

    # Calculate running accuracy
    for i in range(trials):
        is_correct = (predictions[i] == correct_choices[i]).item()
        recent_correct.append(is_correct)
        accuracies.append(sum(recent_correct) / len(recent_correct))
        
        # Track context switches
        if i > 0 and i % trials_per_context == 0:
            context_switches.append(i)


    if plotting:
        # Plot results
        plt.figure(figsize=(15, 5))
        plt.plot(accuracies, 'b-', label='Running Accuracy')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Chance Level')

        # Add context switch lines
        for switch in context_switches:
            plt.axvline(x=switch, color='g', alpha=0.3, linestyle='--')

        plt.title('Model Training Performance with Context Switches')
        plt.xlabel('Trial Number')
        plt.ylabel(f'Accuracy (Last {window_size} Trials)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add context labels
        for i in range(len(context_switches) + 1):
            start = context_switches[i-1] if i > 0 else 0
            end = context_switches[i] if i < len(context_switches) else trials
            mid = (start + end) // 2
            context = conditions_window[mid]['context'].capitalize()
            plt.text(mid, 1.05, context, ha='center')

        plt.ylim(0, 1.1)
        plt.show()

    return accuracies, context_switches


def moving_window2(net, trials = 1000, window_size = 40, trials_per_context = 100):
    """
    Track and plot model performance through a series of context switches 
    in training mode

    Args:
        net: the model
        trials: total number of trials:
        window_size: the number of trials used to calculate accuracy (last X trials)
        trials_per_context: number of trials per context before a switch

    """

    
    import torch.nn as nn
    import Tmaze_UniToMulti as TMUM
    import matplotlib.pyplot as plt
    import seaborn as sns
    import importlib
    from collections import deque


    net.train()

    u_train, z_train, mask_train, conditions_window = TMUM.TmazeEnv2.generate_trials(trials, alpha=0.5, sigma_in=0.1, baseline=0.2, n_coh=8, 
                                                                  modality='multi', n_t=75, trials_per_context=trials_per_context)
    
    device = next(net.parameters()).device
    u_train = u_train.to(device)

    accuracies = []
    recent_correct = deque(maxlen=window_size)
    context_switches = []

    outputs, _ = net(u_train)
    decisions = outputs[:, -1, :]
    predictions = (decisions[:, 0] > decisions[:, 1]).float()
    correct_choices = torch.tensor([1 if c['correct_choice'] == 1 else 0 
                                  for c in conditions_window], 
                                  device=device)

    # Calculate running accuracy
    for i in range(trials):
        is_correct = (predictions[i] == correct_choices[i]).item()
        recent_correct.append(is_correct)
        accuracies.append(sum(recent_correct) / len(recent_correct))
        
        # Track context switches
        if i > 0 and i % trials_per_context == 0:
            context_switches.append(i)

    return accuracies, context_switches





