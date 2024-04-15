from typing import Any
from functools import lru_cache
from time import sleep
import threading
import cv2
import numpy
import onnxruntime
from tqdm import tqdm

import facefusion.globals
from facefusion import process_manager, wording
from facefusion.typing import VisionFrame, ModelSet, Fps
from facefusion.execution import apply_execution_provider_options
from facefusion.vision import get_video_frame, count_video_frame_total, read_image, detect_video_fps
from facefusion.filesystem import resolve_relative_path, is_file
from facefusion.download import conditional_download

CONTENT_ANALYSER = None
THREAD_LOCK : threading.Lock = threading.Lock()
MODELS : ModelSet =\
{
	'open_nsfw':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/open_nsfw.onnx',
		'path': resolve_relative_path('../.assets/models/open_nsfw.onnx')
	}
}
PROBABILITY_LIMIT = 1.80
RATE_LIMIT = 900
STREAM_COUNTER = 0


def get_content_analyser() -> Any:
	global CONTENT_ANALYSER

	with THREAD_LOCK:
		while process_manager.is_checking():
			sleep(0.5)
		if CONTENT_ANALYSER is None:
			model_path = MODELS.get('open_nsfw').get('path')
			CONTENT_ANALYSER = onnxruntime.InferenceSession(model_path, providers = apply_execution_provider_options(facefusion.globals.execution_providers))
	return CONTENT_ANALYSER


def clear_content_analyser() -> None:
	global CONTENT_ANALYSER

	CONTENT_ANALYSER = None


def pre_check() -> bool:
	download_directory_path = resolve_relative_path('../.assets/models')
	model_url = MODELS.get('open_nsfw').get('url')
	model_path = MODELS.get('open_nsfw').get('path')

	if not facefusion.globals.skip_download:
		process_manager.check()
		conditional_download(download_directory_path, [ model_url ])
		process_manager.end()
	return is_file(model_path)


def analyse_stream(vision_frame : VisionFrame, video_fps : Fps) -> bool:
	return None


def analyse_frame(vision_frame : VisionFrame) -> bool:
	return None


def prepare_frame(vision_frame : VisionFrame) -> VisionFrame:
	vision_frame = cv2.resize(vision_frame, (224, 224)).astype(numpy.float32)
	vision_frame -= numpy.array([ 104, 117, 123 ]).astype(numpy.float32)
	vision_frame = numpy.expand_dims(vision_frame, axis = 0)
	return vision_frame


@lru_cache(maxsize = None)
def analyse_image(image_path : str) -> bool:
	return None


@lru_cache(maxsize = None)
def analyse_video(video_path : str, start_frame : int, end_frame : int) -> bool:
	return None
