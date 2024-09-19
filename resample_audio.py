import os
from glob import glob

import librosa
import soundfile as sf


###
# IMPORTANT:
#   - This script will remove original audio files.
###
def resample_audio_to_16k(audio_file, output_file, target_sample_rate=16000):
    audio_data, sample_rate = librosa.load(audio_file, sr=None)

    if sample_rate != target_sample_rate:
        audio_data = librosa.resample(
            audio_data, orig_sr=sample_rate, target_sr=target_sample_rate
        )

    sf.write(output_file, audio_data, target_sample_rate)
    os.remove(audio_file)
    print(f"Resampled audio saved to {output_file}")


def resample_audio_to_16k_in_folder(file_name, folder_path, target_sample_rate=16000):
    for idx, audio_file in enumerate(glob(os.path.join(folder_path, "*.wav"))):
        output_file = os.path.join(
            os.path.dirname(audio_file), f"{file_name}_{idx}.wav"
        )
        resample_audio_to_16k(audio_file, output_file, target_sample_rate)


if __name__ == "__main__":
    # resample_audio_to_16k_in_folder("mini_speech_commands/heymax")
    resample_audio_to_16k_in_folder("max", "C:/Users/Hawteen/Desktop/max")
