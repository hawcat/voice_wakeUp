import glob

import librosa
import numpy as np
import soundfile as sf


def process_audio(input_path, output_path, target_duration, silence_thresh=-20):
    # 1. 加载音频文件
    audio, sr = librosa.load(input_path, sr=None)

    # 2. 识别音频中的非静音片段
    non_silent_intervals = librosa.effects.split(audio, top_db=-silence_thresh)

    # 3. 提取所有非静音片段并合并
    non_silent_audio = np.concatenate(
        [audio[start:end] for start, end in non_silent_intervals]
    )

    # 4. 计算目标长度对应的样本数
    target_length = int(target_duration * sr)

    # 5. 计算所需填充的静音长度
    if len(non_silent_audio) >= target_length:
        # 如果非静音片段大于或等于目标长度，则裁剪到目标长度
        trimmed_audio = non_silent_audio[:target_length]
        sf.write(output_path, trimmed_audio, sr)
    else:
        # 需要填充的静音长度
        silence_padding = target_length - len(non_silent_audio)
        pad_left = silence_padding // 2
        pad_right = silence_padding - pad_left

        trimmed_audio = np.pad(non_silent_audio, (0, silence_padding), mode="constant")
        output_path = output_path.replace(".wav", "_left.wav")
        sf.write(output_path, trimmed_audio, sr)
        trimmed_audio = np.pad(non_silent_audio, (silence_padding, 0), mode="constant")
        output_path = output_path.replace("_left.wav", "_right.wav")
        sf.write(output_path, trimmed_audio, sr)
        trimmed_audio = np.pad(non_silent_audio, (pad_left, pad_right), mode="constant")
        output_path = output_path.replace("_right.wav", "_central.wav")
        sf.write(output_path, trimmed_audio, sr)


for i in glob.glob("speech_commands_v0.02/max/*.wav"):
    print(f"Processing {i}...")
    output_file = i.replace(
        "speech_commands_v0.02/max", "speech_commands_v0.02/max_trimed"
    )

    process_audio(i, output_file, 1.0)

# inputfile = "trigger_example/4_16khz.wav"
# outputfile = "trigger_example/4_16khz_central.wav"
# process_audio(inputfile, outputfile, 1.0)
