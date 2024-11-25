import numpy as np
import sounddevice as sd
import librosa
import sys
from scipy.signal import butter, filtfilt
import tensorflow as tf  # For CNN model inference (ensure you have a trained model)
import queue

# Parameters
target_sample_rate = 8000
n_fft = 2048
hop_length = 512
n_mels = 128
clip_duration = 5  # Duration of each clip in seconds
target_width = 128

low_cutoff = 93.75  # Low-pass cut-off frequency in Hz
high_cutoff = 2500  # High-pass cut-off frequency in Hz
filter_order = 4    # Order of the filter

# Butterworth band-pass filter
def butter_bandpass(lowcut, highcut, sr, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Applying band-pass filter to an audio signal
def bandpass_filter(data, lowcut, highcut, sr, order=4):
    b, a = butter_bandpass(lowcut, highcut, sr, order=order)
    y = filtfilt(b, a, data)  # Apply the filter to the data
    return y

# Preprocess the audio
def preprocess_audio(audio, sample_rate):
    # Apply bandpass filter
    audio_filtered = bandpass_filter(audio, low_cutoff, high_cutoff, sample_rate, filter_order)

    # Normalize the audio
    audio_filtered = audio_filtered / np.max(np.abs(audio_filtered))

    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_filtered, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Normalize the spectrogram
    log_mel_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.mean()) / log_mel_spectrogram.std()

    # Reshape for CNN (height, width, channels)
    log_mel_spectrogram = log_mel_spectrogram.reshape((n_mels, target_width, 1))

    return log_mel_spectrogram

# Perform inference using a CNN model
def perform_inference(model, spectrogram):
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
    prediction = model.predict(spectrogram)  # Perform inference
    return prediction

# Real-time audio recording callback
def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)

    # Flatten the incoming audio to process as a 1D array
    audio_data = indata.flatten()

    # Preprocess the audio
    preprocessed_audio = preprocess_audio(audio_data, target_sample_rate)

    # Perform inference on the processed audio
    prediction = perform_inference(model, preprocessed_audio)

    # Output the prediction (you can process this output as needed)
    print("Prediction:", prediction)

# Main function to record and process audio in real-time
def main():
    global model
    # Load the trained CNN model (ensure you have a model file, e.g., 'model.h5')
    model = tf.keras.models.load_model('your_cnn_model.h5')

    # Create a stream to record audio in real-time
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=target_sample_rate, blocksize=int(target_sample_rate * clip_duration)):
        print(f"Recording in real-time. Press Ctrl+C to stop...")
        while True:
            # This loop allows continuous real-time audio processing.
            # The actual audio processing happens inside the callback function.
            pass

# Run the main function
if __name__ == "__main__":
    main()
