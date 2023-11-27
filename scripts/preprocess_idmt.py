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
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from nnAudio import features
from omegaconf import DictConfig, OmegaConf
from scipy.io import wavfile
from torchaudio.functional import pitch_shift
from rich.progress import track

import logging
logger = logging.getLogger(__name__)


DATA_PATH = "./data"
IDMT_PATH = os.path.join(DATA_PATH, "raw", "IDMT")
AUDIO_PATH = os.path.join(IDMT_PATH, "audio")
ANNOTATION_PATH = os.path.join(IDMT_PATH, "annotation")


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
                self.feature_extractor = nnAudio.features.CQT1992v2(
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


def load_frets(xml, start_sec, end_sec, pitch_step=0):
    if pitch_step != 0:
        raise NotImplementedError
    
    frets = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

    events = xml.findall(".//event")
    for event in events:
        onset_sec = float(event.find("onsetSec").text)
        pitch = float(event.find("pitch").text)
        offset_sec = float(event.find("offsetSec").text)
        excitation_style = event.find("excitationStyle").text
        expression_style = event.find("expressionStyle").text
        fret_number = float(event.find("fretNumber").text)+pitch_step
        string_number = int(event.find("stringNumber").text)-1
        if string_number > 5:
            logger.warning(f"Unsupport string number: {string_number}")
            return

        if start_sec <= onset_sec <= end_sec:
            frets[string_number].append(fret_number)

    return frets


def process(anno_name, cfg, skip_features=False):
    feature_extractor = FeatureExtractor(cfg)
    data = []
    
    audio_file_name = anno_name + ".wav"
    file_path = os.path.join(ANNOTATION_PATH, anno_name + ".xml")
    audio_path = os.path.join(AUDIO_PATH, audio_file_name)
    info = torchaudio.info(audio_path)
    num_frames = info.num_frames
    sample_rate = info.sample_rate
    duration = int(num_frames / sample_rate)
    xml = ET.parse(file_path).getroot()
    
    for start_sec in np.arange(
        start=0,
        stop=duration - cfg.audio.audio_load_sec + cfg.audio.slide_window_sec,
        step=cfg.audio.slide_window_sec,
    ):
        for pitch_step in (cfg.audio.audio_augmentation.pitch_shift_steps or [0]):
            end_sec = start_sec + cfg.audio.audio_load_sec
            audio_load_sec = cfg.audio.audio_load_sec
            frets = load_frets(xml, start_sec, end_sec, pitch_step)
            if frets is None:
                continue

            feature_path = None
            if cfg.audio.features_dir:
                feature_path = os.path.join(
                    cfg.audio.features_dir,
                    anno_name + f"-{round(start_sec, 2)}_"
                    f"{round(start_sec+audio_load_sec, 2)}"
                    f"-pitch_{pitch_step}.pt",
                )

            if not skip_features and cfg.audio.segments_dir:
                segment_path = os.path.join(
                    cfg.audio.segments_dir,
                    anno_name + f"-{round(start_sec, 2)}_"
                    f"{round(start_sec+audio_load_sec, 2)}"
                    f"-pitch_{pitch_step}.wav",
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
                    anno_name,
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

    os.makedirs(os.path.join(cfg.dataset.features_dir), exist_ok=True)
    if cfg.dataset.segments_dir is not None:
        os.makedirs(os.path.join(cfg.dataset.segments_dir), exist_ok=True)

    file_names = list(filter(lambda p: p.endswith("xml"), os.listdir(ANNOTATION_PATH)))
    logger.info(f"Found {len(file_names)} files")
    file_names = [os.path.splitext(p)[0] for p in file_names]

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

    if cfg.num_workers <= 1:
        for file_name in track(file_names):
            data = process(file_name, cfg)
            if data is not None:  # BUG: Data should not be None, but for some reason this is happening
                dataset = [*dataset, *data]
            else:
                logger.error(f"Error processing file: {file_name} - [BUG] <data> should not be None, but for some reason this is happening")
    else:
        executor = ProcessPoolExecutor(cfg.num_workers)
        futures = [executor.submit(process, file_name, cfg) for file_name in file_names]

        for future in track(as_completed(futures), total=len(file_names)):
            if data is not None:
                dataset = [*dataset, *data]
            else:
                logger.error(f"Error processing file: {file_name} - [BUG] <data> should not be None, but for some reason this is happening")
            dataset = [*dataset, *data]

    random.shuffle(dataset)

    with open(os.path.join(DATA_PATH, f"idmt.csv"), "w") as data_csv:
        csv_writer = csv.writer(data_csv, delimiter=';')
        csv_writer.writerow(header)
        csv_writer.writerows(dataset)


if __name__ == "__main__":
    main()
