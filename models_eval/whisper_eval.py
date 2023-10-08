import time

import whisper

model = whisper.load_model("base")

X = 10

run_count: int = 0

start_time = time.time()
while run_count < X:
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("./audios/ahmad.mp3")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    # print the recognized text
    # print(result.text)

    print(f"whisper  --  Run {run_count} completed!", flush=True)
    run_count += 1


end_time = time.time()
print(
    f"whisper  --  Total time taken: {end_time - start_time} seconds for {X} runs",
    flush=True,
)

