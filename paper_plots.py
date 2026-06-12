import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import numpy as np
import librosa.display

def create_publication_quality_plot(mag_gt, mag_prd, wav_gt, wav_prd, data_idx, save_dir='./paper_plot', sample_rate=22050):
    # Set up publication-quality parameters
    plt.style.use('default')
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['axes.titleweight'] = 'bold'
    rcParams['axes.labelweight'] = 'bold'
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 12
    rcParams['figure.titlesize'] = 16
    rcParams['figure.figsize'] = (20, 8)  # Increased width for better stretching
    rcParams['savefig.dpi'] = 300
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    
    # Define colors for channels
    left_channel_color = '#1f77b4'  # Blue
    right_channel_color = '#ff7f0e'  # Orange
    
    # Create figure with 2 rows and 3 columns (changed from 2x4 to 2x3)
    fig = plt.figure(figsize=(20, 8))
    
    # Create a custom grid to control widths
    # Format is (height_ratios, width_ratios)
    # Making first column wider than the other two
    gs = plt.GridSpec(2, 3, figure=fig, width_ratios=[1.0, 0.8, 0.8], wspace=0.2, hspace=0.3)
    
    # Calculate time axis (x-axis) for waveforms
    duration = wav_gt.shape[1] / sample_rate
    time_axis = np.linspace(0, duration, wav_gt.shape[1])
    
    # 1. Ground Truth Magnitude Spectrogram - using the custom grid
    ax_gt_spec = fig.add_subplot(gs[0, 0])
    img1 = librosa.display.specshow(
        librosa.amplitude_to_db(mag_gt[0]), 
        y_axis='log', 
        x_axis='time',
        sr=sample_rate,
        ax=ax_gt_spec,
        cmap='viridis'
    )
    cbar1 = plt.colorbar(img1, ax=ax_gt_spec, format='%+2.0f dB')
    cbar1.set_label('Magnitude (dB)', fontweight='bold')
    ax_gt_spec.set_title('GT Mag Spectrogram', fontweight='bold')  # Shortened title
    ax_gt_spec.set_xlabel('Time (s)', fontweight='bold')
    ax_gt_spec.set_ylabel('Frequency (Hz)', fontweight='bold')
    
    # Set tick label font weight to bold manually
    for tick in ax_gt_spec.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax_gt_spec.get_yticklabels():
        tick.set_fontweight('bold')
    
    # 2. Ground Truth Waveform - Left Channel
    ax_gt_wav_left = fig.add_subplot(gs[0, 1])
    ax_gt_wav_left.plot(time_axis, wav_gt[0], color=left_channel_color, linewidth=1.0)
    ax_gt_wav_left.set_title('GT Waveform - Left', fontweight='bold')  # Shortened title
    ax_gt_wav_left.set_xlabel('Time (s)', fontweight='bold')
    ax_gt_wav_left.set_ylabel('Amplitude', fontweight='bold')
    ax_gt_wav_left.set_xlim([0, duration])
    y_max = max(abs(wav_gt[0].min()), abs(wav_gt[0].max())) * 1.1
    ax_gt_wav_left.set_ylim([-y_max, y_max])
    ax_gt_wav_left.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax_gt_wav_left.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    
    # Set tick label font weight to bold manually
    for tick in ax_gt_wav_left.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax_gt_wav_left.get_yticklabels():
        tick.set_fontweight('bold')
    
    # 3. Ground Truth Waveform - Right Channel
    ax_gt_wav_right = fig.add_subplot(gs[0, 2])
    ax_gt_wav_right.plot(time_axis, wav_gt[1], color=right_channel_color, linewidth=1.0)
    ax_gt_wav_right.set_title('GT Waveform - Right', fontweight='bold')  # Shortened title
    ax_gt_wav_right.set_xlabel('Time (s)', fontweight='bold')
    ax_gt_wav_right.set_ylabel('Amplitude', fontweight='bold')
    ax_gt_wav_right.set_xlim([0, duration])
    y_max = max(abs(wav_gt[1].min()), abs(wav_gt[1].max())) * 1.1
    ax_gt_wav_right.set_ylim([-y_max, y_max])
    ax_gt_wav_right.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax_gt_wav_right.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    
    # Set tick label font weight to bold manually
    for tick in ax_gt_wav_right.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax_gt_wav_right.get_yticklabels():
        tick.set_fontweight('bold')
    
    # 4. Predicted Magnitude Spectrogram
    ax_pred_spec = fig.add_subplot(gs[1, 0])
    img2 = librosa.display.specshow(
        librosa.amplitude_to_db(mag_prd[0]), 
        y_axis='log', 
        x_axis='time',
        sr=sample_rate,
        ax=ax_pred_spec,
        cmap='viridis'
    )
    cbar2 = plt.colorbar(img2, ax=ax_pred_spec, format='%+2.0f dB')
    cbar2.set_label('Magnitude (dB)', fontweight='bold')
    ax_pred_spec.set_title('Pred Mag Spectrogram', fontweight='bold')  # Shortened title
    ax_pred_spec.set_xlabel('Time (s)', fontweight='bold')
    ax_pred_spec.set_ylabel('Frequency (Hz)', fontweight='bold')
    
    # Set tick label font weight to bold manually
    for tick in ax_pred_spec.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax_pred_spec.get_yticklabels():
        tick.set_fontweight('bold')
    
    # 5. Predicted Waveform - Left Channel
    ax_pred_wav_left = fig.add_subplot(gs[1, 1])
    ax_pred_wav_left.plot(time_axis, wav_prd[0], color=left_channel_color, linewidth=1.0)
    ax_pred_wav_left.set_title('Pred Waveform - Left', fontweight='bold')  # Shortened title
    ax_pred_wav_left.set_xlabel('Time (s)', fontweight='bold')
    ax_pred_wav_left.set_ylabel('Amplitude', fontweight='bold')
    ax_pred_wav_left.set_xlim([0, duration])
    y_max = max(abs(wav_prd[0].min()), abs(wav_prd[0].max())) * 1.1
    ax_pred_wav_left.set_ylim([-y_max, y_max])
    ax_pred_wav_left.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax_pred_wav_left.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    
    # Set tick label font weight to bold manually
    for tick in ax_pred_wav_left.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax_pred_wav_left.get_yticklabels():
        tick.set_fontweight('bold')
    
    # 6. Predicted Waveform - Right Channel
    ax_pred_wav_right = fig.add_subplot(gs[1, 2])
    ax_pred_wav_right.plot(time_axis, wav_prd[1], color=right_channel_color, linewidth=1.0)
    ax_pred_wav_right.set_title('Pred Waveform - Right', fontweight='bold')  # Shortened title
    ax_pred_wav_right.set_xlabel('Time (s)', fontweight='bold')
    ax_pred_wav_right.set_ylabel('Amplitude', fontweight='bold')
    ax_pred_wav_right.set_xlim([0, duration])
    y_max = max(abs(wav_prd[1].min()), abs(wav_prd[1].max())) * 1.1
    ax_pred_wav_right.set_ylim([-y_max, y_max])
    ax_pred_wav_right.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax_pred_wav_right.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    
    # Set tick label font weight to bold manually
    for tick in ax_pred_wav_right.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax_pred_wav_right.get_yticklabels():
        tick.set_fontweight('bold')
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save with high resolution
    plt.savefig(os.path.join(save_dir, f'comparison_batch_{data_idx}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'comparison_batch_{data_idx}.pdf'), format='pdf', bbox_inches='tight')
    
    plt.close()