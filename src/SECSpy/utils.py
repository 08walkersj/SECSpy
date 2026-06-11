"""General utility helpers used by SECSpy examples and workflows.

This module contains small helpers that do not belong to the core SECS
inversion code: image-sequence animation, simple command-line prompting, and
date/filename conversion.
"""

import os

import numpy as np

from SECSpy.SECS import ArgumentError
import functools

def dates2npdate(func):
    @functools.wraps(func)
    def wrapper(date, *args):
        import pandas as pd
        from datetime import datetime
        if isinstance(date, np.datetime64):
            pass
        elif isinstance(date, pd.Timestamp):
            date=date.to_numpy()
        elif isinstance(date, datetime):
            date=np.datetime64(date)
        else:
            raise ArgumentError(f'date type yet not known, update code or change format. Current type: {type(date)}')
        return func(date, *args)
    return wrapper
@dates2npdate
def date2filename(date):
    import re
    return '_'.join('_'.join(re.split('   |-|:', f'{date.astype("datetime64[m]").tolist()}')).split(' '))
def filename2date(filename):
    date=np.empty(3).astype(str)
    time=np.empty(3).astype(str)
    date[0], date[1], date[2], time[0], time[1], time[2]= filename.split('_')
    return np.datetime64('-'.join(date)+'T'+':'.join(time))


def get_image_size(image_path):
    """
    Return the pixel dimensions of an image file.

    Parameters
    ----------
    image_path : str or path-like
        Path to an image readable by Pillow.

    Returns
    -------
    tuple of int
        Image size as ``(width, height)`` in pixels.
    """
    from PIL import Image

    with Image.open(image_path) as img:
        return img.size


def _audio_args(audio_path, audio_volume=1.0, audio_offset=0.0, loop_audio=False):
    """
    Build the ffmpeg command arguments needed to add an audio input.

    Parameters
    ----------
    audio_path : str or path-like
        Path to the audio file.
    audio_volume : float, optional
        Audio gain multiplier. ``1.0`` leaves volume unchanged, ``0.5`` halves
        it, and ``2.0`` doubles it.
    audio_offset : float, optional
        Time shift in seconds. Positive values delay the audio; negative values
        trim the start so audio begins earlier in the output.
    loop_audio : bool, optional
        If True, loop the audio input. The final ffmpeg command should also use
        ``-shortest`` so encoding stops when the video ends.

    Returns
    -------
    list of str
        ffmpeg argument list fragment for the audio input, filters, and codec.
    """
    args = []
    if loop_audio:
        # -stream_loop applies only to the next input, so it must precede -i.
        args += ['-stream_loop', '-1']
    args += ['-i', audio_path]

    afilters = []
    if audio_volume != 1.0:
        afilters.append(f'volume={audio_volume}')
    if audio_offset > 0:
        delay_ms = int(audio_offset * 1000)
        # adelay expects a per-channel delay spec, separated with '|'.
        afilters.append(f'adelay={delay_ms}|{delay_ms}')
    elif audio_offset < 0:
        # Drop the initial segment, then reset timestamps to start at zero.
        afilters.append(f'atrim=start={abs(audio_offset)},asetpts=PTS-STARTPTS')

    if afilters:
        args += ['-af', ','.join(afilters)]

    args += ['-c:a', 'aac', '-b:a', '192k']

    return args


def create_video_from_images(
    folder_path,
    prefix='',
    filename='output',
    gif_filename=None,
    num_digits=2,
    start_number=1,
    fps=5,
    audio_path=None,
    audio_volume=1.0,
    audio_offset=0.0,
    loop_audio=False,
    create_gif=True,
):
    """
    Create an MP4 video, and optionally a GIF, from numbered PNG files.

    The function expects an image sequence named with a shared prefix and a
    zero-padded integer counter, for example ``frame01.png``, ``frame02.png``,
    ... when ``prefix='frame'`` and ``num_digits=2``. ffmpeg is used for the
    actual encoding.

    Parameters
    ----------
    folder_path : str or path-like
        Folder containing the PNG image sequence. The output files are written
        to the same folder.
    prefix : str, optional
        Filename prefix before the numeric counter.
    filename : str, optional
        Base name or path for the generated ``.mp4``. If only a name is
        supplied, the video is written inside ``folder_path``. If a path is
        supplied, that path is used. The ``.mp4`` suffix is added when omitted.
    gif_filename : str, optional
        Base name or path for the generated ``.gif``. If omitted, the GIF uses
        the same path stem as ``filename`` with a ``.gif`` suffix.
    num_digits : int, optional
        Number of zero-padded digits in the image sequence counter.
    start_number : int, optional
        Number of the first frame in the image sequence.
    fps : int or float, optional
        Output frame rate.
    audio_path : str or path-like, optional
        Optional audio file to include in the MP4.
    audio_volume : float, optional
        Audio gain multiplier passed to ffmpeg.
    audio_offset : float, optional
        Audio time offset in seconds. Positive values delay the audio; negative
        values trim the beginning of the audio.
    loop_audio : bool, optional
        If True, loop the audio and stop encoding at the end of the video.
    create_gif : bool, optional
        If True, also create a GIF version using ``convert_video_to_gif``.

    Raises
    ------
    RuntimeError
        Raised when ffmpeg returns a non-zero exit code.

    Returns
    -------
    None
    """
    import shlex
    import subprocess

    def resolve_output_path(name, default_folder, suffix):
        path = os.fspath(name)
        root, ext = os.path.splitext(path)
        if not ext:
            path = path + suffix
        elif ext.lower() != suffix:
            path = root + suffix
        if os.path.dirname(path):
            return path
        return os.path.join(default_folder, path)


    # Use the first PNG to derive a fixed output size for the whole sequence.
    image_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith('.png'))
    if not image_files:
        print(f"No images found in folder: {folder_path}")
        return

    first_image = os.path.join(folder_path, image_files[0])
    width, height = get_image_size(first_image)
    def even(x): return x if x % 2 == 0 else x - 1
    width  = even(width)
    height = even(height)
    scale_filter = f"scale={width}:{height}:flags=lanczos"

    input_pattern = os.path.join(folder_path, prefix + f"%0{num_digits}d.png")
    output_video = resolve_output_path(filename, folder_path, '.mp4')
    output_folder = os.path.dirname(output_video)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-stats", "-y",

        # ffmpeg requires the sequence framerate before the image input.
        "-framerate", str(fps),

        # Input 0 is the image sequence.
        "-start_number", str(start_number),
        "-i", input_pattern,
    ]

    if audio_path:
        if loop_audio:
            # -stream_loop applies to the next -i only.
            cmd += ["-stream_loop", "-1"]

        # Input 1 is the optional audio track.
        cmd += ["-i", audio_path]

    cmd += [
        "-vf", scale_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
    ]

    # Add audio filters if present
    if audio_path:
        af = []
        if audio_volume != 1.0:
            af.append(f"volume={audio_volume}")
        if audio_offset > 0:
            af.append(f"adelay={int(audio_offset*1000)}|{int(audio_offset*1000)}")
        elif audio_offset < 0:
            af.append(f"atrim=start={abs(audio_offset)},asetpts=PTS-STARTPTS")

        if af:
            cmd += ["-af", ",".join(af)]

        cmd += ["-c:a", "aac", "-b:a", "192k"]

        # Explicit stream mapping avoids ffmpeg choosing an unintended stream.
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]

    cmd += [output_video]

    print("FFmpeg CMD:", " ".join(shlex.quote(c) for c in cmd))

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(p.stderr)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed")

    print("✅ Created video:", output_video)
    if create_gif:
        # Convert video to GIF using two-pass palette method
        if gif_filename is None:
            gif_path = os.path.splitext(output_video)[0] + '.gif'
        else:
            gif_path = resolve_output_path(gif_filename, folder_path, '.gif')
        convert_video_to_gif(output_video, gif_path, width, height, fps=fps)


def convert_video_to_gif(video_path, gif_path, width, height, fps=5):
    """
    Convert a video file to a GIF using ffmpeg's palette workflow.

    A temporary palette image is generated first, then reused when creating the
    GIF. This usually gives a better-looking GIF than a single-pass conversion.

    Parameters
    ----------
    video_path : str or path-like
        Path to the source video.
    gif_path : str or path-like
        Output path for the GIF.
    width, height : int
        GIF dimensions in pixels.
    fps : int or float, optional
        GIF frame rate.
    Returns
    -------
    None
    """
    import subprocess

    gif_output = os.fspath(gif_path)
    gif_folder = os.path.dirname(gif_output) or '.'
    os.makedirs(gif_folder, exist_ok=True)
    palette_path = os.path.join(gif_folder, 'palette.png')
    scale_filter = f'scale={width}:{height}:flags=lanczos'

    # Step 1: Generate palette
    palettegen_command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vf', f'fps={fps},{scale_filter},palettegen',
        palette_path
    ]
    subprocess.run(palettegen_command)
    
    # Step 2: Create GIF using palette
    gif_command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', palette_path,
        '-filter_complex', f'fps={fps},{scale_filter}[x];[x][1:v]paletteuse',
        gif_output
    ]
    subprocess.run(gif_command)
    if os.path.exists(palette_path):
        os.remove(palette_path)
    print(f"Created GIF: {gif_output}")


def process_all_folders(base_folder, **create_video_kwargs):
    """
    Run ``create_video_from_images`` for every folder under a base directory.

    Parameters
    ----------
    base_folder : str or path-like
        Root directory to walk recursively.
    **create_video_kwargs
        Keyword arguments forwarded to ``create_video_from_images``.

    Returns
    -------
    None
    """
    for root, dirs, files in os.walk(base_folder):
        create_video_from_images(root, **create_video_kwargs)


def validinput(inputstr, positive_answer, negative_answer):
    """
    Prompt until the user enters one of two accepted answers.

    Parameters
    ----------
    inputstr : str
        Prompt displayed to the user.
    positive_answer : str
        Input value that should return ``True``.
    negative_answer : str
        Input value that should return ``False``.

    Returns
    -------
    bool
        True for ``positive_answer`` and False for ``negative_answer``.
    """
    answer= input(inputstr+'\n')
    if answer==positive_answer:
        return True
    elif answer== negative_answer:
        return False
    else:
        print('Invalid response should be either '+ str(positive_answer)+ ' or ' +str(negative_answer))
        return validinput(inputstr, positive_answer, negative_answer)


def dates2npdate(func):
    """
    Decorate a function so its first argument is converted to ``numpy.datetime64``.

    Accepted input types are ``numpy.datetime64``, ``pandas.Timestamp``, and
    ``datetime.datetime``. pandas is not imported unless the object already
    looks like a pandas timestamp, keeping the common NumPy/datetime path free
    of an optional dependency. Other types raise ``ArgumentError``.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(date, *args):
        from datetime import datetime

        if isinstance(date, np.datetime64):
            pass
        elif isinstance(date, datetime):
            date=np.datetime64(date)
        elif type(date).__module__.startswith('pandas.') and hasattr(date, 'to_numpy'):
            date=date.to_numpy()
        else:
            raise ArgumentError(f'date type yet not known, update code or change format. Current type: {type(date)}')
        return func(date, *args)
    return wrapper


@dates2npdate
def date2filename(date):
    """
    Convert a date/time value to the timestamp format used in SECSpy paths.

    Parameters
    ----------
    date : numpy.datetime64, pandas.Timestamp, or datetime.datetime
        Date/time value to encode. Resolution is truncated to minutes.

    Returns
    -------
    str
        Timestamp string formatted as ``YYYY_MM_DD_HH_MM_SS``.
    """
    import re

    return '_'.join('_'.join(re.split('   |-|:', f'{date.astype("datetime64[m]").tolist()}')).split(' '))


def filename2date(filename):
    """
    Parse a SECSpy timestamp filename back to ``numpy.datetime64``.

    Parameters
    ----------
    filename : str
        Timestamp string formatted as ``YYYY_MM_DD_HH_MM_SS``.

    Returns
    -------
    numpy.datetime64
        Parsed datetime value.
    """
    date=np.empty(3).astype(str)
    time=np.empty(3).astype(str)
    date[0], date[1], date[2], time[0], time[1], time[2]= filename.split('_')
    return np.datetime64('-'.join(date)+'T'+':'.join(time))
