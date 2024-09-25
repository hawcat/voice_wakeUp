import glob

import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def remove_silence(audio, silence_threshold=-30, min_silence_len=100):
    """
    去除音频中的静音部分
    :param audio: AudioSegment 对象
    :param silence_threshold: 静音阈值（dB）
    :param min_silence_len: 最小静音长度（毫秒）
    :return: 去除静音后的 AudioSegment 对象
    """
    nonsilent_chunks = detect_nonsilent(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_threshold
    )
    if not nonsilent_chunks:
        return AudioSegment.empty()

    # 合并非静音部分
    nonsilent_audio = AudioSegment.empty()
    for start, end in nonsilent_chunks:
        nonsilent_audio += audio[start:end]

    return nonsilent_audio


def pad_audio_to_target_length(audio, target_length_samples, sample_rate):
    """
    将音频居中并补充静音至目标采样点数
    :param audio: AudioSegment 对象
    :param target_length_samples: 目标采样点数
    :param sample_rate: 采样率（Hz）
    :return: 填充后的 AudioSegment 对象
    """
    current_length_samples = len(audio.get_array_of_samples())
    if current_length_samples >= target_length_samples:
        return audio

    silence_length_samples = target_length_samples - current_length_samples
    left_silence_samples = silence_length_samples // 2
    right_silence_samples = silence_length_samples - left_silence_samples

    left_silence = AudioSegment.silent(
        duration=left_silence_samples * 1000 / sample_rate
    )
    right_silence = AudioSegment.silent(
        duration=right_silence_samples * 1000 / sample_rate
    )

    padded_audio = left_silence + audio + right_silence

    # 确保精确匹配目标采样点数
    if len(padded_audio.get_array_of_samples()) > target_length_samples:
        padded_audio = padded_audio[:target_length_samples]
    elif len(padded_audio.get_array_of_samples()) < target_length_samples:
        padded_audio += AudioSegment.silent(
            duration=(target_length_samples - len(padded_audio.get_array_of_samples()))
            * 1000
            / sample_rate
        )

    return padded_audio


def process_audio(
    input_file,
    output_file,
    target_length_samples,
    sample_rate,
    silence_threshold=-40,
    min_silence_len=100,
):
    """
    处理音频文件：去除静音，居中并补充静音至目标采样点数
    :param input_file: 输入音频文件路径
    :param output_file: 输出音频文件路径
    :param target_length_samples: 目标采样点数
    :param sample_rate: 采样率（Hz）
    :param silence_threshold: 静音阈值（dB）
    :param min_silence_len: 最小静音长度（毫秒）
    """
    # 加载音频文件
    audio = AudioSegment.from_file(input_file)

    # 去除静音部分
    nonsilent_audio = remove_silence(audio, silence_threshold, min_silence_len)

    # 居中并补充静音至目标采样点数
    padded_audio = pad_audio_to_target_length(
        nonsilent_audio, target_length_samples, sample_rate
    )

    # 导出处理后的音频
    padded_audio.export(output_file, format="wav")


# for i in glob.glob("dataset/max/*.wav"):
#     print(f"Processing {i}...")
#     output_file = i.replace("dataset/max", "dataset/max_central")
#     target_length_samples = 16000  # 目标采样点数为16000
#     sample_rate = 16000  # 采样率为16000 Hz

#     process_audio(i, output_file, target_length_samples, sample_rate)

input_file = "trigger_example/4_16khz.wav"
output_file = "trigger_example/4_16khz_central.wav"
target_length_samples = 16000  # 目标采样点数为16000
sample_rate = 16000  # 采样率为16000 Hz
process_audio(input_file, output_file, target_length_samples, sample_rate)
