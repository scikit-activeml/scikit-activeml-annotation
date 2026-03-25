from io import BytesIO

from dash import dcc
import dash_player
import dash_mantine_components as dmc

import numpy as np
from plotly import graph_objects as go

import librosa
import soundfile
 
from skactiveml_annotation.core import api

from ._model import AudioDataDisplaySetting

from .. import ids


AUDIO_AUTOPLAY = { 'type': ids.DATA_PRESENTATION_INPUT, 'property': 'checked', 'modality': 'audio', 'index': 'autoplay' }
AUDIO_LOOP_INPUT = { 'type': ids.DATA_PRESENTATION_INPUT, 'property': 'checked', 'modality': 'audio', 'index': 'loop' }
AUDIO_PLAYBACK_RATE_INPUT = { 'type': ids.DATA_PRESENTATION_INPUT, 'property': 'value', 'modality': 'audio', 'index': 'playback_rate' }

def display(audio_data_path, audio_display_setting, format ="WAV"):
    """
    Creates a Dash Mantine AudioPlayer component from a local WAV file.
    
    Parameters:
    - audio_data_path: str, path to your local .wav file
    - audio_display_setting: dict, optional settings (e.g., width)
    
    Returns:
    - dmc.Center containing the AudioPlayer
    """

    # Load data from audio file as Pulse Code Modulation (PCM) timeseries 
    # into numpy array.
    # librosa uses soundfile and audiofile as a backup so all their file formats
    # are supported
    time_series, sample_rate = librosa.load(audio_data_path, sr=None) # Use native sampling rate

    # Convert raw timeseries data into raw in memory representation of a wav file
    wav_file_bytes_buffer = BytesIO()
    soundfile.write(file=wav_file_bytes_buffer, data=time_series, samplerate=sample_rate, format=format)
        
    # Pretty much all browsers support wav format: https://caniuse.com/wav
    inline_wave_file_url = api.file_buffer_to_inline_data_url(
        wav_file_bytes_buffer,
        mime=f"audio/{format.lower()}"
    )

    player = dash_player.DashPlayer(
        url=inline_wave_file_url,
        controls=True,
        loop=audio_display_setting.loop,
        playbackRate=audio_display_setting.playback_rate,
        playing=audio_display_setting.autoplay,
        height=50,
    )

    fig = _create_log_mel_power_spectogramm_fig(time_series, sample_rate)

    # Create the Dash Mantine AudioPlayer
    return dmc.Center(
        dmc.Stack(
            [
                dcc.Graph(figure=fig),
                player
            ]
        ),
        m="xl"
    )


def presentation_settings():
    default_audio_setting = AudioDataDisplaySetting()

    return (
        dmc.Stack(
            [
                dmc.Checkbox(
                    id=AUDIO_LOOP_INPUT,
                    label="Looping",
                    checked=default_audio_setting.loop,
                    persistence=str(AUDIO_LOOP_INPUT),
                    persistence_type='session'
                ),

                dmc.Checkbox(
                    id=AUDIO_AUTOPLAY,
                    label="Autoplay",
                    checked=default_audio_setting.autoplay,
                    persistence=str(AUDIO_AUTOPLAY),
                    persistence_type='session'
                ),
            
                dmc.NumberInput(
                    id=AUDIO_PLAYBACK_RATE_INPUT,
                    min=0.2,
                    max=35,
                    step=0.1,
                    clampBehavior='strict',
                    hideControls=False,
                    decimalScale=2,
                    label="Playback Rate",
                    placeholder=str(default_audio_setting.playback_rate),
                    value=default_audio_setting.playback_rate,
                    allowNegative=False,
                    # w='35%',
                    persistence=str(AUDIO_PLAYBACK_RATE_INPUT),
                    persistence_type='session'
                ),
            ],
            align='start'
        )
    )


def _create_log_mel_power_spectogramm_fig(
    time_series,
    sample_rate,
    n_fft=4096,
    hop_length=256,
    n_mels=128,
    fmin=80,
    fmax=8000,
):
    fmin = fmin
    # fmax should never exeed Nyhilist limit
    fmax = min(sample_rate / 2, fmax) 

    # Frequency axis
    S = librosa.feature.melspectrogram(
        y=time_series, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels, 
        center=True,
        fmin=fmin,
        fmax=fmax
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    # Center of each mel frequency bin in regular Hz
    freqs = librosa.mel_frequencies(n_mels=S.shape[0], fmin=fmin, fmax=fmax)
    # Mel frequencies are not equy distant. Lower frequencies get stretched apart
    # and higher frequencies get squished closer together
    mel_freq = librosa.hz_to_mel(freqs)

    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sample_rate, hop_length=hop_length)

    tick_idxs = np.arange(0, len(freqs), 32)
    tickvals = mel_freq[tick_idxs]
    ticktext = [f"{int(freqs[i])}" for i in tick_idxs]

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=S_db,
            x=times,
            y=mel_freq,
            colorscale="magma",
            colorbar=dict(title="dB"),
            zmin=S_db.min(),
            zmax=S_db.max(),
        )
    )

    axis_args = dict(
        showticklabels=True,
        ticks="outside",
        ticklen=6,
        tickwidth=1.5,
        showline=True,
        mirror=True,
        tickcolor="black",
        linecolor="black",
    )

    # Configure axes
    fig.update_layout(
        title_text="Log Mel Power Spectrogram",
        title_x = 0.5,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        xaxis=dict(**axis_args),
        yaxis=dict(
            tickmode="array", # Specify that it should use tickvals to determine tick positions
            tickvals=tickvals,
            ticktext=ticktext,
            **axis_args
        )
    )

    return fig
