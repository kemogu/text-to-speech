# train_tts.py

from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
from TTS.config.tts_config import TTSConfig
from TTS.tts.models import Tacotron2
from TTS.bin.train_tts import prepare_and_train

# 1) Dataset config
dataset_config = BaseDatasetConfig(
    name="dataset",
    meta_file="dataset/train.csv",
    path="dataset/wavs"
)

# 2) Audio config
audio_config = BaseAudioConfig(
    sample_rate=22050,
    trim_silence=True
)

# 3) TTS config
tts_config = TTSConfig(
    model=Tacotron2,
    use_pretrained=True,
    pretrained_model_path="tts_models/en/ljspeech/tacotron2-DDC",
    audio=audio_config,
    datasets=[dataset_config],
    batch_size=16,
    eval_batch_size=16,
    epochs=1000,
    learning_rate=1e-4
)

# 4) Launch training
if __name__ == "__main__":
    prepare_and_train(tts_config)
