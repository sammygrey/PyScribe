import os
import numpy as np
import speech_recognition as sr

from transformers import WhisperProcessor, AutoProcessor
from ctranslate2 import StorageView, models, converters
from datetime import datetime, timedelta, timezone
from queue import Queue
from time import sleep
from sys import platform
from silero_vad import load_silero_vad, get_speech_timestamps

def main():

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = "audio=Microphone (2- USB Audio CODEC )"
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)
        
    model_name = "openai/whisper-large-v3-turbo"

    processor =  WhisperProcessor.from_pretrained(model_name) #AutoProcessor.from_pretrained(mode_id) if you have modified the model
    tokenizer = processor.tokenizer
    
    #try: 
        #converter = converters.TransformersConverter(model_name)
        #if you have stronger hardware feel free to forgo the quantization, this reduces memory usage for large models on lesser hardware
        #converter.convert(output_dir="whisper-turbo-ct2", quantization="int8_float16", force=True)
    #except:
       #print("Model has already been converted to CTranslate2 format; conversion not necessary.")

    converted_model = "whisper-turbo-ct2"
    audio_model = models.Whisper(converted_model, device="cuda")
    
    print("Model Loaded.\n")
    
    record_timeout = 2
    phrase_timeout = 3

    transcription = ['']
    logits = []

    with source:
        #recorder.use_silero()
        recorder.adjust_for_ambient_noise(source)


    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)
        
        
    def make_prompt(tokenizer, language="en") -> list[int]:
        prompt = tokenizer.convert_tokens_to_ids(
            [
                "<|startoftranscript|>",
                f"<|{language}|>",
                "<|transcribe|>",
                #"<|notimestamps|>", #uncomment this for no timestamps
            ]
        )
        return prompt


    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Started Recording.\n")

    while True:
        try:
            now = datetime.now(timezone.utc)
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 16 bits
                # If you removed the quantization when moving the model to CT2, change to astype(np.float32)
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float16) / 32768.0
                
                
                # Read the transcription.
                inputs = processor(audio_np, return_tensors="np", sampling_rate=16000)
                features = StorageView.from_array(inputs["input_features"])
                prompt = make_prompt(tokenizer)
                results = audio_model.generate(features, [prompt], return_scores=True) #include_prompt_in_result useful for debugging
                text = processor.decode(results[0].sequences_ids[0])
                logit = results[0].scores

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                    #add logits score to log
                    logits.append(logit)
                    
                else:
                    transcription[-1] = transcription[-1] + text
                    #math to combine logits scores of what are treated as seperate probabilities by the model
                    if len(logits) > 0:
                        logits[-1] = logits[-1] + logit
                    else:
                        logits.append(logit)
                    
                    
                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                # Flush stdout.
                print('', end='', flush=True)
            else:
                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("Stopped Recording.\n")
    print("\n\nTranscription:")
    for line in transcription:
        print(line)
    for logit in logits:
        print(logit)


if __name__ == "__main__":
    main()