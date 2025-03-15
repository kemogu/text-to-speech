from TTS.utils.synthesizer import Synthesizer

model_path = "output/best_model.pth.tar"
config_path = "output/config.json"

synth = Synthesizer(model_path, config_path)
output_wav = synth.tts("Hello world, this is my custom TTS model.")
synth.save_wav(output_wav, "my_tts_output.wav")
