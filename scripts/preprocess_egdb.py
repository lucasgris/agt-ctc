import csv
import os
import random
import warnings

import hydra
import librosa
import nnAudio
import numpy as np
import torch
import torchaudio
import mido
from concurrent.futures import ProcessPoolExecutor, as_completed
from nnAudio import features
from omegaconf import DictConfig, OmegaConf
from scipy.io import wavfile
from torchaudio.functional import pitch_shift
from rich.progress import track
from amt_tools import tools

import logging
logger = logging.getLogger(__name__)

STR_MIDI_DICT = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}

DATA_PATH = "./data"
EGDB_PATH = os.path.join(DATA_PATH, "raw", "EGDB")
AUDIO_PATHS = [
    os.path.join(EGDB_PATH, a)
    for a in [
        "audio_DI",
        "audio_Ftwin",
        "audio_JCjazz",
        "audio_Marshall",
        "audio_Mesa",
        "audio_Plexi",
    ]
]
ANNOTATION_PATH = os.path.join(EGDB_PATH, "audio_label")

REAL_DATA_PATH = os.path.join(EGDB_PATH, "RealData")
REAL_DATA_AUDIO_PATH = os.path.join(REAL_DATA_PATH, "Audio")
REAL_DATA_ANNOTATION_PATH = os.path.join(REAL_DATA_PATH, "Label")


def load_stacked_notes_midi(midi_path):
    """
    Extract MIDI notes spread across strings into a dictionary
    from a MIDI file following the EGDB format.

    Parameters
    ----------
    midi_path : string
      Path to MIDI file to read

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    # Standard tuning is assumed for all tracks in EGDB
    open_tuning = list(librosa.note_to_midi(tools.DEFAULT_GUITAR_TUNING))

    # Initialize a dictionary to hold the notes for each string
    stacked_notes = [tools.notes_to_stacked_notes([], [], p) for p in open_tuning]
    stacked_notes = {k: v for d in stacked_notes for k, v in d.items()}

    # Open the MIDI file
    midi = mido.MidiFile(midi_path)

    # Initialize a counter for the time
    time = 0

    # Initialize an empty list to store MIDI events
    events = []

    # Parse all MIDI messages
    for message in midi:
        # Increment the time
        time += message.time

        # Check if message is a note event (NOTE_ON or NOTE_OFF)
        if "note" in message.type:
            # Determine corresponding string index
            string_idx = 5 - message.channel
            # MIDI offsets can be either NOTE_OFF events or NOTE_ON with zero velocity
            onset = message.velocity > 0 if message.type == "note_on" else False

            # Create a new event detailing the note
            event = dict(time=time, pitch=message.note, onset=onset, string=string_idx)
            # Add note event to MIDI event list
            events.append(event)

    # Loop through all tracked MIDI events
    for i, event in enumerate(events):
        # Ignore note offset events
        if not event["onset"]:
            continue

        # Extract note attributes
        pitch = event["pitch"]
        onset = event["time"]
        string_idx = event["string"]

        # Determine where the corresponding offset occurs by finding the next note event
        # with the same string, clipping at the final frame if no correspondence is found
        offset = next(
            n
            for n in events[i + 1 :]
            if n["string"] == event["string"] or n is events[-1]
        )["time"]

        # Obtain the current collection of pitches and intervals
        pitches, intervals = stacked_notes.pop(open_tuning[string_idx])

        # Append the (nominal) note pitch
        pitches = np.append(pitches, pitch)
        # Append the note interval
        intervals = np.append(intervals, [[onset, offset]], axis=0)

        # Re-insert the pitch-interval pairs into the stacked notes dictionary under the appropriate key
        stacked_notes.update(
            tools.notes_to_stacked_notes(pitches, intervals, open_tuning[string_idx])
        )

    # Re-order keys starting from lowest string and switch to the corresponding note label
    stacked_notes = {
        librosa.midi_to_note(i): stacked_notes[i] for i in sorted(stacked_notes.keys())
    }

    return stacked_notes


class FeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.audio.device
        if cfg.audio.feature_extractor == "nnAudio":
            if cfg.audio.feature == "chroma_stft":
                self.feature_extractor = nnAudio.features.ChromaSTFT(
                    sr=cfg.audio.chroma_stft_params.sr,
                    n_fft=cfg.audio.chroma_stft_params.n_fft,
                    n_chroma=cfg.audio.chroma_stft_params.n_chroma,
                    hop_length=cfg.audio.chroma_stft_params.hop_length,
                ).to(self.device)
            elif cfg.audio.feature == "cqt":
                self.feature_extractor = nnAudio.features.CQT(
                    sr=cfg.audio.cqt_params.sr,
                    hop_length=cfg.audio.cqt_params.hop_length,
                    n_bins=cfg.audio.cqt_params.n_bins,
                    bins_per_octave=cfg.audio.cqt_params.bins_per_octave,
                ).to(self.device)
            elif cfg.audio.feature == "spectrogram":
                self.feature_extractor = nnAudio.features.STFT(
                    n_fft=cfg.audio.spectrogram_params.n_fft,
                    win_length=cfg.audio.spectrogram_params.win_length,
                    hop_length=cfg.audio.spectrogram_params.hop_length,
                ).to(self.device)
            elif cfg.audio.feature == "melspectrogram":
                self.feature_extractor = nnAudio.Spectrogram.MelSpectrogram(
                    sample_rate=cfg.audio.sr,
                    n_fft=cfg.audio.melspectrogram_params.n_fft,
                    win_length=cfg.audio.melspectrogram_params.win_length,
                    hop_length=cfg.audio.melspectrogram_params.hop_length,
                    n_mels=cfg.audio.melspectrogram_params.n_mels,
                    f_min=cfg.audio.melspectrogram_params.f_min,
                    f_max=cfg.audio.melspectrogram_params.f_max,
                ).to(self.device)
            else:
                raise ValueError(f"Invalid feature: {cfg.audio.feature}")
        elif cfg.audio.feature_extractor == "torchaudio":
            if cfg.audio.feature == "chroma_stft":
                self.feature_extractor = torchaudio.transforms.ChromaSTFT(
                    sr=cfg.audio.chroma_stft_params.sr,
                    n_fft=cfg.audio.chroma_stft_params.n_fft,
                    n_chroma=cfg.audio.chroma_stft_params.n_chroma,
                    hop_length=cfg.audio.chroma_stft_params.hop_length,
                ).to(self.device)
            elif cfg.audio.feature == "cqt":
                self.feature_extractor = torchaudio.transforms.CQT(
                    sr=cfg.audio.cqt_params.sr,
                    hop_length=cfg.audio.cqt_params.hop_length,
                    n_bins=cfg.audio.cqt_params.n_bins,
                    bins_per_octave=cfg.audio.cqt_params.bins_per_octave,
                ).to(self.device)
            elif cfg.audio.feature == "spectrogram":
                self.feature_extractor = torchaudio.transforms.Spectrogram(
                    n_fft=cfg.audio.spectrogram_params.n_fft,
                    pad_mode="reflect",
                    center=True,
                    win_length=cfg.audio.spectrogram_params.win_length,
                    hop_length=cfg.audio.spectrogram_params.hop_length,
                ).to(self.device)
            elif cfg.audio.feature == "melspectrogram":
                self.feature_extractor = torchaudio.transforms.MelSpectrogram(
                    sample_rate=cfg.audio.sr,
                    center=True,
                    n_fft=cfg.audio.melspectrogram_params.n_fft,
                    win_length=cfg.audio.melspectrogram_params.win_length,
                    hop_length=cfg.audio.melspectrogram_params.hop_length,
                    n_mels=cfg.audio.melspectrogram_params.n_mels,
                    f_min=cfg.audio.melspectrogram_params.f_min,
                    f_max=cfg.audio.melspectrogram_params.f_max,
                ).to(self.device)
            else:
                raise ValueError(f"Invalid feature: {cfg.audio.feature}")
        elif cfg.audio.feature_extractor == "librosa":
            self.feature_extractor = None
        else:
            raise ValueError(
                f"Invalid feature extractor: {cfg.audio.feature_extractor}"
            )

    def __call__(self, x):
        if self.cfg.audio.feature_extractor == "librosa":
            if self.cfg.audio.feature == "chroma_stft":
                feature = np.abs(
                    librosa.feature.chroma_stft(
                        y=np.array(x),
                        sr=self.cfg.audio.chroma_stft_params.sr,
                        n_fft=self.cfg.audio.chroma_stft_params.n_fft,
                        n_chroma=self.cfg.audio.chroma_stft_params.n_chroma,
                        hop_length=self.cfg.audio.chroma_stft_params.hop_length,
                    )
                )
            elif self.cfg.audio.feature == "cqt":
                feature = np.abs(
                    librosa.cqt(
                        y=np.array(x),
                        sr=self.cfg.audio.cqt_params.sr,
                        hop_length=self.cfg.audio.cqt_params.hop_length,
                        n_bins=self.cfg.audio.cqt_params.n_bins,
                        bins_per_octave=self.cfg.audio.cqt_params.bins_per_octave,
                    )
                )
            elif self.cfg.audio.feature == "spectrogram":
                feature = np.abs(
                    librosa.stft(
                        y=np.array(x),
                        n_fft=self.cfg.audio.spectrogram_params.n_fft,
                        win_length=self.cfg.audio.spectrogram_params.win_length,
                        hop_length=self.cfg.audio.spectrogram_params.hop_length,
                    )
                )
            elif self.cfg.audio.feature == "melspectrogram":
                feature = np.abs(
                    librosa.feature.melspectrogram(
                        y=np.array(x),
                        sr=self.cfg.audio.sr,
                        n_fft=self.cfg.audio.melspectrogram_params.n_fft,
                        win_length=self.cfg.audio.melspectrogram_params.win_length,
                        hop_length=self.cfg.audio.melspectrogram_params.hop_length,
                        n_mels=self.cfg.audio.melspectrogram_params.n_mels,
                        f_min=self.cfg.audio.melspectrogram_params.f_min,
                        f_max=self.cfg.audio.melspectrogram_params.f_max,
                    )
                )
            else:
                raise ValueError(f"Invalid feature: {self.cfg.audio.feature}")
            feature = torch.from_numpy(feature)
            feature = feature.unsqueeze(0)
        else:
            feature = self.feature_extractor(x.to(self.device))
        return feature


def extract_features(
    cfg,
    feature_extractor,
    audio_path,
    feature_path,
    start_sec,
    audio_load_sec,
    pitch_step=0,
    onset_feature_path=None,
    segment_path=None,
    force_reprocess=False,
    device="cpu",
):
    if not force_reprocess:
        if os.path.isfile(feature_path):
            feature = torch.load(feature_path)
            feature_size = feature.size()
            logger.info(
                f"Segment already processed: audio_path={audio_path} start_sec={start_sec} audio_load_sec={audio_load_sec} feature_path={feature_path} feature_size={feature_size}"
            )
            return feature_size

    if cfg.audio.loader == "torchaudio":
        if segment_path is not None and os.path.isfile(segment_path):
            x, sr = torchaudio.load(segment_path, normalize=True)
        else:
            sr = torchaudio.info(audio_path).sample_rate
            x, sr = torchaudio.load(
                audio_path,
                normalize=True,
                frame_offset=int(start_sec * sr),
                num_frames=int(audio_load_sec * sr),
            )

        if cfg.audio.resample and sr != cfg.audio.sr:
            if cfg.audio.loader == "torchaudio":
                x = torchaudio.transforms.Resample(orig_freq=sr, new_freq=cfg.audio.sr)(
                    x
                )

        x = torch.mean(x, dim=0)
        x = x.to(device)
    elif cfg.audio.loader == "librosa":
        if segment_path is not None and os.path.isfile(segment_path):
            x, sr = librosa.load(
                segment_path, mono=True, sr=cfg.audio.sr if cfg.audio.resample else None
            )
        else:
            x, sr = librosa.load(
                audio_path,
                mono=True,
                sr=cfg.audio.sr if cfg.audio.resample else None,
                offset=start_sec,
                duration=audio_load_sec,
            )
        x = torch.from_numpy(x).to(device)
    else:
        raise ValueError(f"Invalid loader: {cfg.audio.loader}")

    if pitch_step != 0:
        x = pitch_shift(x.cuda(), sr, pitch_step).cpu()
    feature = feature_extractor(x)

    if segment_path and not os.path.isfile(segment_path):
        wavfile.write(segment_path, sr, np.array(x.cpu()))

    torch.save(feature, feature_path)

    if onset_feature_path is not None:
        warnings.warn("Onset feature extraction is really slow")
        x, sr = librosa.load(audio_path, sr=cfg.audio.sr)
        onset_frames = librosa.onset.onset_detect(
            y=x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1
        )
        onset_frames = np.array(onset_frames)
        onset_times = librosa.frames_to_time(onset_frames)
        onset_times = np.array(onset_times)
        np.save(onset_feature_path, onset_times)

    return feature.size()


def load_frets(midi_path, start_sec, end_sec, pitch_step=0):
    if pitch_step != 0:
        raise NotImplementedError

    frets = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

    stacked_notes = load_stacked_notes_midi(midi_path)
    for s, s_name in enumerate(stacked_notes):
        pitches = stacked_notes[s_name][0]
        intervals = stacked_notes[s_name][1]
        for value, interval in zip(pitches, intervals):
            onset_sec = interval[0]
            duration = interval[1] - interval[0]
            if value is not None:
                fret_pos = int(round(value - STR_MIDI_DICT[s]))
                if not 0 <= fret_pos <= 25:
                    continue

                if start_sec <= onset_sec <= end_sec:
                    frets[s].append(fret_pos)

    return frets


def process(annotation_path, audio_dirs, cfg, skip_features=False):
    feature_extractor = FeatureExtractor(cfg)
    data = []

    audio_load_sec = cfg.audio.audio_load_sec
    for audio_dir in audio_dirs:
        timbre = os.path.basename(audio_dir)
        audio_file_name = (
            os.path.splitext(os.path.basename(annotation_path))[0] + ".wav"
        )
        audio_path = os.path.join(audio_dir, audio_file_name)

        assert os.path.isfile(audio_path), f"File not found: {audio_path}"
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate

        for start_sec in np.arange(
            start=0,
            stop=duration - cfg.audio.audio_load_sec + cfg.audio.slide_window_sec,
            step=cfg.audio.slide_window_sec,
        ):
            for pitch_step in (cfg.audio.audio_augmentation.pitch_shift_steps or [0]):
                end_sec = start_sec + cfg.audio.audio_load_sec
                audio_load_sec = cfg.audio.audio_load_sec
                frets = load_frets(annotation_path, start_sec, end_sec, pitch_step)
                if frets is None:
                    continue

                feature_path = None
                if cfg.dataset.features_dir:
                    feature_path = os.path.join(
                        cfg.dataset.features_dir,
                        audio_file_name + f"-{timbre}_" + f"-{round(start_sec, 2)}_"
                        f"{round(start_sec+audio_load_sec, 2)}.pt",
                    )

                if not skip_features and cfg.dataset.segments_dir:
                    segment_path = os.path.join(
                        cfg.dataset.segments_dir,
                        audio_file_name + f"-{timbre}_" + f"-{round(start_sec, 2)}_"
                        f"{round(start_sec+audio_load_sec, 2)}.wav",
                    )
                else:
                    segment_path = None

                feature_size = None
                if not skip_features and cfg.dataset.features_dir:
                    try:
                        feature_size = extract_features(
                            cfg=cfg,
                            feature_extractor=feature_extractor,
                            audio_path=audio_path,
                            feature_path=feature_path,
                            start_sec=start_sec,
                            audio_load_sec=audio_load_sec,
                            segment_path=segment_path,
                            pitch_step=pitch_step,
                        )
                    except Exception as e:
                        logger.error(e)
                        raise e
                    
                if feature_size is None and os.path.isfile(feature_path):
                    feature = torch.load(feature_path)
                    feature_size = feature.size()
                    del feature

                data.append(
                    [
                        annotation_path,
                        round(start_sec, 2),
                        round(start_sec + audio_load_sec, 2),
                        round(audio_load_sec, 2),
                        pitch_step,
                        frets[0],
                        frets[1],
                        frets[2],
                        frets[3],
                        frets[4],
                        frets[5],
                        str(feature_size).replace(',', ' '),
                        os.path.basename(feature_path),
                        os.path.basename(segment_path),
                        feature_path,
                        segment_path,
                    ]
                )

        return data


@hydra.main(version_base=None, config_path="../configs", config_name="audio_config")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    os.makedirs(os.path.join(cfg.dataset.features_dir), exist_ok=True)
    if cfg.dataset.segments_dir is not None:
        os.makedirs(os.path.join(cfg.dataset.segments_dir), exist_ok=True)

    real_annotation_paths = [
        os.path.join(REAL_DATA_ANNOTATION_PATH, p)
        for p in list(
            filter(lambda p: p.endswith(".mid"), os.listdir(REAL_DATA_ANNOTATION_PATH))
        )
    ]
    annotation_paths = [
        os.path.join(ANNOTATION_PATH, p)
        for p in list(
            filter(lambda p: p.endswith(".midi"), os.listdir(ANNOTATION_PATH))
        )
    ]
    logger.info(f"Found {len(annotation_paths)+len(real_annotation_paths)} files")

    header = [
        "file_name",
        "start_sec",
        "end_sec",
        "duration",
        "pitch_step",
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "feature_size",
        "feature_filename",
        "segment_filename",
        "feature_path",
    ]

    dataset = []
    dataset_real = []
    dataset_rendered = []

    if cfg.num_workers <= 1:
        for annotation_path in track(annotation_paths):
            data = process(annotation_path, AUDIO_PATHS, cfg)
            dataset = [*dataset, *data]
            dataset_rendered = [*dataset_rendered, *data]
        for annotation_path in track(real_annotation_paths):
            data = process(annotation_path, [REAL_DATA_AUDIO_PATH], cfg)
            dataset = [*dataset, *data]
            dataset_real = [*dataset_real, *data]
    else:
        executor = ProcessPoolExecutor(cfg.num_workers)
        futures = [
            executor.submit(process, annotation_path, AUDIO_PATHS, cfg)
            for annotation_path in annotation_paths
        ]
        for future in track(
            as_completed(futures), total=len(annotation_paths), leave=False
        ):
            data = future.result()
            dataset = [*dataset, *data]
            dataset_rendered = [*dataset_rendered, *data]

        futures = [
            executor.submit(process, annotation_path, [REAL_DATA_AUDIO_PATH], cfg)
            for annotation_path in real_annotation_paths
        ]
        for future in track(
            as_completed(futures), total=len(real_annotation_paths), leave=False
        ):
            data = future.result()
            dataset = [*dataset, *data]
            dataset_real = [*dataset_real, *data]

    random.shuffle(dataset)

    with open(os.path.join(DATA_PATH, f"egdb.csv"), "w") as data_csv:
        csv_writer = csv.writer(data_csv, delimiter=';')
        csv_writer.writerow(header)
        csv_writer.writerows(dataset)
    with open(os.path.join(DATA_PATH, f"egdb_rendered.csv"), "w") as data_csv:
        csv_writer = csv.writer(data_csv, delimiter=';')
        csv_writer.writerow(header)
        csv_writer.writerows(dataset_rendered)
    with open(os.path.join(DATA_PATH, f"egdb_real.csv"), "w") as data_csv:
        csv_writer = csv.writer(data_csv, delimiter=';')
        csv_writer.writerow(header)
        csv_writer.writerows(dataset_real)


if __name__ == "__main__":
    main()
