import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np

def plot_zscore_signal(time_x, zscore_signal, baseline_end_idx, title="Z-score of 5HT2A Signal"):
    baseline_zscore = np.mean(zscore_signal[:baseline_end_idx])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time_x, zscore_signal, linewidth=2, color='blue', label='Z-score (serotonin - ISOS)')
    ax1.axhline(y=baseline_zscore, color='red', linestyle='--', label='Baseline')
    
    ax1.set_ylabel('Z-score')
    ax1.set_xlabel('Seconds')
    ax1.set_title(title)
    ax1.legend()
    
    fig.tight_layout()
    plt.show()

def plot_overlay(velocity_data, serotonin_data, injection_relative_time, stop_time_seconds, start_time_seconds):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=velocity_data['Time (seconds)'], y=velocity_data['Smoothed Velocity (cm/s)'], mode='lines', name='Mobility'))
    fig.add_trace(go.Scatter(x=serotonin_data['Time (seconds)'], y=serotonin_data['Z-score'], mode='lines', name='Z-score (serotonin - ISOS)', yaxis='y2'))
    fig.add_vline(x=injection_relative_time, line_width=2, line_dash="dash", line_color="red", annotation_text="Injection", annotation_position="top")
    
    fig.update_layout(
        title='Overlay of Mobility and Serotonin Z-score Over Time',
        xaxis_title='Time (s)',
        yaxis_title='Mobility',
        xaxis=dict(range=[0, stop_time_seconds - start_time_seconds]),
        yaxis=dict(title='Velocity (cm/s)', tickmode='array'),
        yaxis2=dict(title='Z-score', overlaying='y', side='right'),
        legend=dict(x=0, y=1.1, orientation='h')
    )
    
    fig.show()

def plot_velocities(velocities, labels, title="Average Velocity Comparison", xlabel="Files", ylabel="Average Velocity", colors=None):
    plt.figure(figsize=(8, 6))
    plt.bar(labels, velocities, color=colors if colors else ['blue', 'orange'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_distances(distances, labels, title="Total Distance Traveled Comparison", xlabel="Files", ylabel="Total Distance Traveled", colors=None):
    plt.figure(figsize=(8, 6))
    plt.bar(labels, distances, color=colors if colors else ['blue', 'orange'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
