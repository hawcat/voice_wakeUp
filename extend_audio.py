import os
from glob import glob

import librosa
import numpy as np
import soundfile as sf


def extend_audio_to_target(audio_file, output_file, target_duration=1.0):
    # 加载音频文件，保留原始采样率
    audio_data, sample_rate = librosa.load(audio_file, sr=None)

    # 计算目标音频长度
    target_length = int(sample_rate * target_duration)

    # 如果音频已经是目标长度或更长，则直接截断
    if len(audio_data) >= target_length:
        audio_data = audio_data[:target_length]
    else:
        # 计算需要填充的静音长度
        pad_length = target_length - len(audio_data)

        # 在前后添加静音，使音频居中
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left

        # 使用 numpy 进行填充，前后加上零
        audio_data = np.pad(audio_data, (pad_left, pad_right), "constant")

    # 保存处理后的音频
    sf.write(output_file, audio_data, sample_rate)
    os.remove(audio_file)
    print(f"Processed audio saved to {output_file}")


def extend_audio_to_target_in_folder(folder_path, target_duration=2.0):
    for audio_file in glob(os.path.join(folder_path, "*.wav")):
        output_file = os.path.join(
            folder_path, f"{os.path.basename(audio_file)}_extended.wav"
        )
        extend_audio_to_target(audio_file, output_file, target_duration)


if __name__ == "__main__":
    extend_audio_to_target_in_folder("mini_speech_commands/yes")
