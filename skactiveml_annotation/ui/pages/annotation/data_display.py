# TODO Todo Make it so not all dependencies for all modalities have to be installed
from pathlib import Path
from io import BytesIO

from dash import dcc
import dash_player
import dash_mantine_components as dmc

import numpy as np
from plotly import graph_objects as go

# Audio libs
import librosa
import soundfile
# Image libs
from PIL import Image as pil_image
 
from skactiveml_annotation.core.data_display_model import (
    TextDataDisplaySetting,
    ImageDataDisplaySetting,
)

from skactiveml_annotation.core import api

# TODO make components out of these.
def create_image_display(
    path_to_img: Path, 
    image_display_setting: ImageDataDisplaySetting, 
    dpr: float
):
    image = pil_image.open(path_to_img).convert("RGB")

    rescale_factor = image_display_setting.rescale_factor
    w = int(image.width * rescale_factor)
    h = int(image.height * rescale_factor)

    image = image.resize(
        (w, h),
        resample=image_display_setting.resampling_method
        # reducing_gap=4  # Only in effect when downscaling
    )

    # image = image.filter(ImageFilter.UnsharpMask(
    #     radius=200,  # Effects the size of the edges to be enhanced. How wide the edge rims become
    #                   # Smaller Radius enhances smaller-scale detail
    #     percent=100,  # Amount is listed as a percentage and controlls how much constrast is added at edges
    #     threshold=3  # Controls how far adjacent tonal values have to be before the filter does anything
    #     )
    # )
    # image = ImageEnhance.Sharpness(image).enhance(50)

    # image.show()

    # Account for screen dpr to prevent the browser from resizing the image again to avoid artifacts.
    w = int(w / dpr)
    h = int(h / dpr)

    ml = 0
    mt = 0
    mr = 0
    mb = 0
    fig = go.Figure(
        data=go.Image(
            # TODO instead of Inline URI server file on demand?
            source=pil_image_to_inline_data_url(image, format="PNG"),
            # z=image,
        ),
        layout=go.Layout(
            width=w,
            height=h,
            margin=dict(l=ml, r=mr, t=mt, b=mb, pad=0)
        )
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return (
        dcc.Graph(
            figure=fig,
            responsive=False,
            style={
                # 'border':'blue solid 2px',
                # imageRendering:'pixelated'
            },
            config={
                # autosizeable=False,
                'scrollZoom':True,
                'doubleClick':'reset',
                'displaylogo':False,
                'modeBarButtonsToRemove':['toImage'],
            },
        ),
        w,
        h
    )


def pil_image_to_inline_data_url(pil_image: pil_image.Image, format: str ="PNG") -> str:
    """
    Convert a PIL Image to a base64-encoded string.

    Args:
        img:        PIL.Image.Image instance to encode.
        fmt:        Format to save the image in (e.g. "PNG", "JPEG").

    Returns:
        Data URL for the base64 encoded image.
    """
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    mime = f"image/{format.lower()}"
    return api.file_buffer_to_inline_data_url(buffer, mime)


# TODO: Introduce different modules for different modalities
def create_text_display(path: Path, text_display_setting: TextDataDisplaySetting):
    if not path.exists():
        raise ValueError(f"Cannot load text data from path: {path}")

    text_data = path.read_text(encoding="utf-8").strip()

    return \
        dmc.ScrollArea(
            dmc.Box(
                dcc.Markdown(
                    text_data,
                    style={
                        "margin": "0",
                        "padding": "0",
                        "fontSize": f"{text_display_setting.font_size}px",
                        "fontFamily": "'Consolas', 'Courier New', 'Lucida Console', monospace",
                        "fontVariantLigatures": "none",  # disable ligatures

                        "lineHeight": text_display_setting.line_height,
                        # Preserve line breaks but wrap long lines
                        "whiteSpace": "pre-line",  # allow wrapping (default is normal)
                        "wordBreak": "normal",  # break long words if needed
                        "overflowWrap": "normal", # ensures text doesn't overflow
                        # Width of the Text Container
                        "overflowX": "hidden", # no horizontal scrollbar
                        
                        # 'border':'blue dotted 2px',
                    }
                ),
                py=1,
                style={
                    # TODO hardcoded width
                    "width": "46vw",        # restrict component width
                    "overflowX": "hidden",      # no horizontal scrollbar

                    # "border": "2px solid green",  # Debug border
                },
            ),
            type='auto',
            offsetScrollbars='y',
            styles=dict(
                viewport={
                    'maxHeight': '60vh',
                    # 'border':'brown dashed 3px',
                },
            )
        )


def create_audio_display(audio_data_path, audio_display_setting, format ="WAV"):
    """
    Creates a Dash Mantine AudioPlayer component from a local WAV file.
    
    Parameters:
    - audio_data_path: str, path to your local .wav file
    - audio_display_setting: dict, optional settings (e.g., width)
    
    Returns:
    - dmc.Center containing the AudioPlayer
    """

    print("Path", audio_data_path)

    # Load data from audio file as Pulse Code Modulation (PCM) timeseries 
    # into numpy array.
    # librosa uses soundfile and audiofile as a backup so all their file formats
    # are supported
    time_series, sample_rate = librosa.load(audio_data_path, sr=None) # Use native sampling rate

    duration_in_sec = len(time_series) * (1 / sample_rate)
    print("\n The duration of the sample in sec is:", duration_in_sec)
    print("Sample Rate:", sample_rate)
    print()

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
        height=50,
    )

    fig = create_log_mel_power_spectogramm_fig(time_series, sample_rate)

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


def create_log_mel_power_spectogramm_fig(
    time_series, 
    sample_rate, 
    n_fft=4096,
    hop_length=256, 
    n_mels=128,
    fmin=80,
    fmax=8000
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
