from typing import Optional, Any, List
from functools import lru_cache
import numpy
import scipy

from facefusion.filesystem import is_audio
from facefusion.ffmpeg import read_audio_buffer
from facefusion.typing import Fps, Audio, AudioFrame, Spectrogram, MelFilter
from facefusion.voice_extractor import batch_extract_voice


@lru_cache(maxsize = 128)
def read_static_audio(audio_path : str, fps : Fps) -> Optional[List[AudioFrame]]:
	return read_audio(audio_path, fps)


def read_audio(audio_path : str, fps : Fps) -> Optional[List[AudioFrame]]:
	if is_audio(audio_path):
		audio_buffer = read_audio_buffer(audio_path, 16000, 2)
		audio = numpy.frombuffer(audio_buffer, dtype = numpy.int16).reshape(-1, 2)
		audio = normalize_audio(audio)
		audio = filter_audio(audio, -0.97)
		spectrogram = create_spectrogram(audio, 16000, 80, 800, 55.0, 7600.0)
		audio_frames = extract_audio_frames(spectrogram, 80, 16, fps)
		return audio_frames
	return None


@lru_cache(maxsize = 128)
def read_static_voice(audio_path : str, fps : Fps) -> Optional[List[AudioFrame]]:
	return read_voice(audio_path, fps)


def read_voice(audio_path : str, fps : Fps) -> Optional[List[AudioFrame]]:
	if is_audio(audio_path):
		audio_buffer = read_audio_buffer(audio_path, 16000, 2)
		audio = numpy.frombuffer(audio_buffer, dtype = numpy.int16).reshape(-1, 2)
		audio = batch_extract_voice(audio, 1024 ** 3, 0.75)
		audio = normalize_audio(audio)
		audio = filter_audio(audio, -0.97)
		spectrogram = create_spectrogram(audio, 16000, 80, 800, 55.0, 7600.0)
		audio_frames = extract_audio_frames(spectrogram, 80, 16, fps)
		return audio_frames
	return None


def get_audio_frame(audio_path : str, fps : Fps, frame_number : int = 0) -> Optional[AudioFrame]:
	if is_audio(audio_path):
		audio_frames = read_static_audio(audio_path, fps)
		if frame_number in range(len(audio_frames)):
			return audio_frames[frame_number]
	return None


def get_voice_frame(audio_path : str, fps : Fps, frame_number : int = 0) -> Optional[AudioFrame]:
	if is_audio(audio_path):
		voice_frames = read_static_voice(audio_path, fps)
		if frame_number in range(len(voice_frames)):
			return voice_frames[frame_number]
	return None


def create_empty_audio_frame() -> AudioFrame:
	audio_frame = numpy.zeros((80, 16)).astype(numpy.int16)
	return audio_frame


def normalize_audio(audio : numpy.ndarray[Any, Any]) -> Audio:
	if audio.ndim > 1:
		audio = numpy.mean(audio, axis = 1)
	audio = audio / numpy.max(numpy.abs(audio), axis = 0)
	return audio


def filter_audio(audio : Audio, filter_coefficient : float) -> Audio:
	audio = scipy.signal.lfilter([ 1.0, filter_coefficient ], [ 1.0 ], audio)
	return audio


def convert_hertz_to_mel(hertz : float) -> float:
	return 2595 * numpy.log10(1 + hertz / 700)


def convert_mel_to_hertz(mel : numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
	return 700 * (10 ** (mel / 2595) - 1)


def create_mel_filter(sample_rate : int, filter_total : int, filter_size : int, frequency_minimum : float, frequency_maximum : float) -> MelFilter:
	mel_filter = numpy.zeros((filter_total, filter_size // 2 + 1))
	mel_bins = numpy.linspace(convert_hertz_to_mel(frequency_minimum), convert_hertz_to_mel(frequency_maximum), filter_total + 2)
	indices = numpy.floor((filter_size + 1) * convert_mel_to_hertz(mel_bins) / sample_rate).astype(numpy.int16)

	for index in range(filter_total):
		start = indices[index]
		end = indices[index + 1]
		mel_filter[index, start:end] = scipy.signal.windows.triang(end - start)
	return mel_filter


def create_spectrogram(audio : Audio, sample_rate : int, filter_total : int, filter_size : int, frequency_minimum : float, frequency_maximum : float) -> Spectrogram:
	mel_filter = create_mel_filter(sample_rate, filter_total, filter_size, frequency_minimum, frequency_maximum)
	spectrogram = scipy.signal.stft(audio, nperseg = filter_size, noverlap = 600, nfft = filter_size)[2]
	spectrogram = numpy.dot(mel_filter, numpy.abs(spectrogram))
	return spectrogram


def extract_audio_frames(spectrogram : Spectrogram, filter_total : int, audio_frame_step : int, fps : Fps) -> List[AudioFrame]:
	audio_frames = []
	indices = numpy.arange(0, spectrogram.shape[1], filter_total / fps).astype(numpy.int16)
	indices = indices[indices >= audio_frame_step]

	for index in indices:
		start = max(0, index - audio_frame_step)
		audio_frames.append(spectrogram[:, start:index])
	return audio_frames
