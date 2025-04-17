#functions to support LSTM analysis
import numpy as np
import matplotlib.pyplot as plt
import torch

def analyze_lstm_dynamics(net, u, conditions):
    device = next(net.parameters()).device
    u = u.to(device)
    
    with torch.no_grad():
        # Separate trials by context and choice
        visual_trials = [i for i, c in enumerate(conditions) if c['context'] == 'visual']
        audio_trials = [i for i, c in enumerate(conditions) if c['context'] == 'audio']
        
        # Forward pass
        outputs, (final_hidden, final_cell) = net.lstm(u)
        hidden_states = outputs.detach().cpu().numpy()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot averaged hidden states by context
        colors = plt.cm.Set2(np.linspace(0, 1, 5))
        
        # Visual context trials
        if visual_trials:
            visual_mean = np.mean(hidden_states[visual_trials], axis=0)
            for i in range(min(5, visual_mean.shape[1])):
                ax1.plot(visual_mean[:, i], '-', 
                        color=colors[i], 
                        label=f'Unit {i} (Visual)',
                        alpha=0.8,
                        linewidth=2)
        
        # Audio context trials
        if audio_trials:
            audio_mean = np.mean(hidden_states[audio_trials], axis=0)
            for i in range(min(5, audio_mean.shape[1])):
                ax1.plot(audio_mean[:, i], '--',
                        color=colors[i], 
                        label=f'Unit {i} (Audio)',
                        alpha=0.8,
                        linewidth=2)
        
        # Add vertical line for decision point
        decision_point = hidden_states.shape[1] // 2
        ax1.axvline(x=decision_point, color='red', 
                   linestyle=':', label='Decision Point')
        
        ax1.set_title('Hidden State Dynamics by Context')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Activation')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Cell state analysis
        cell_states = final_cell.squeeze().detach().cpu().numpy()
        
        # Plot cell state distributions separated by context
        visual_cells = cell_states[visual_trials]
        audio_cells = cell_states[audio_trials]
        
        labels = ['Visual', 'Audio']
        ax2.violinplot([visual_cells[:, i] for i in range(min(5, cell_states.shape[1]))],
                      positions=np.arange(5)*2)
        ax2.violinplot([audio_cells[:, i] for i in range(min(5, cell_states.shape[1]))],
                      positions=np.arange(5)*2 + 1)
        
        ax2.set_title('Cell State Distributions by Context')
        ax2.set_xlabel('Unit')
        ax2.set_ylabel('Value')
        ax2.set_xticks(np.arange(10))
        ax2.set_xticklabels([f'U{i//2}{l}' for i in range(10) 
                            for l in ['V', 'A'][i%2]])
        
        plt.tight_layout()
        return hidden_states, cell_states


def check_lstm_gradients(net):
    """Check LSTM gradient properties"""
    # Get LSTM parameters
    lstm_params = {name: param for name, param in net.lstm.named_parameters()}
    
    print("LSTM Gradient Statistics:")
    for name, param in lstm_params.items():
        if param.grad is not None:
            grad = param.grad.data
            print(f"\n{name}:")
            print(f"Gradient mean: {grad.mean():.2e}")
            print(f"Gradient std: {grad.std():.2e}")
            print(f"Gradient min: {grad.min():.2e}")
            print(f"Gradient max: {grad.max():.2e}")

def test_temporal_memory(net, sequence_length=50):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    """Test if LSTM can maintain information over time"""
    device = next(net.parameters()).device
    
    # Create test sequence with temporal dependency
    x = torch.zeros((1, sequence_length, net.input_size)).to(device)
    x[0, 0, 0] = 1  # Set initial input
    
    # Run network
    with torch.no_grad():
        outputs, _ = net(x)
    
    # Analyze output correlation with initial input
    outputs_np = outputs.cpu().numpy().squeeze()
    correlation = np.corrcoef(outputs_np[:, 0], x[0, :, 0].cpu().numpy())[0,1]
    
    print(f"Temporal correlation: {correlation:.3f}")
    return correlation