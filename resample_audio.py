import os
from glob import glob

import librosa
import soundfile as sf
from pydub import AudioSegment


def adjust_volume(input_file, output_file, target_dB):
    audio = AudioSegment.from_file(input_file)
    difference = target_dB - audio.dBFS
    adjusted_audio = audio + difference
    adjusted_audio.export(output_file, format="wav")


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
    print(f"Resampled audio saved to {output_file}")


def resample_audio_to_16k_in_folder(folder_path, target_sample_rate=16000):
    if os.path.isfile(folder_path):
        audio_file = folder_path
        output_file = os.path.join(
            os.path.dirname(audio_file),
            f"{os.path.basename(audio_file)[:-4]}_16khz.wav",
        )
        resample_audio_to_16k(audio_file, output_file, target_sample_rate)

    elif os.path.isdir(folder_path):
        for idx, audio_file in enumerate(glob(os.path.join(folder_path, "*.wav"))):
            output_file = os.path.join(
                os.path.dirname(audio_file),
                f"{os.path.basename(audio_file)[:-4]}_16khz.wav",
            )
            resample_audio_to_16k(audio_file, output_file, target_sample_rate)


if __name__ == "__main__":
    # resample_audio_to_16k_in_folder("mini_speech_commands/heymax")
    resample_audio_to_16k_in_folder("trigger_example/max.wav")
