import numpy as np
import sounddevice as sd
import librosa
import sys
from scipy.signal import butter, filtfilt
import tensorflow as tf  # For CNN model inference (ensure you have a trained model)
import time

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
    # Ensure audio length matches the expected duration
    target_length = int(sample_rate * clip_duration)  # Total samples for 5 seconds
    if len(audio) < target_length:
        # Pad with zeros if audio is shorter than target length
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        # Trim if audio is longer than target length
        audio = audio[:target_length]

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

    # Adjust the spectrogram dimensions to match the CNN input
    target_width = 128  # Desired width
    if log_mel_spectrogram.shape[1] < target_width:
        # Pad with zeros if the width is less than target
        pad_width = target_width - log_mel_spectrogram.shape[1]
        log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Trim if the width is greater than target
        log_mel_spectrogram = log_mel_spectrogram[:, :target_width]

    # Reshape for CNN (height, width, channels)
    log_mel_spectrogram = log_mel_spectrogram.reshape((n_mels, target_width, 1))

    return log_mel_spectrogram


# Perform inference using a CNN model
def perform_inference(model, spectrogram):
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
    prediction = model.predict(spectrogram)  # Perform inference
    return prediction

# Main function to record audio, process, and predict
def main():
    global model
    # Load the trained CNN model
    model = tf.keras.models.load_model('C:/Users/ROBVIC/Documents/RPI_Preprocess_Predict/cnn_weevil_full.h5')
    print("Model loaded successfully.")

    # Calculate number of samples for 5 seconds
    total_samples = int(target_sample_rate * clip_duration)

    try:
        while True:
            print("Capturing audio for 5 seconds...")
            # Record audio for 5 seconds
            audio_data = sd.rec(total_samples, samplerate=target_sample_rate, channels=1, dtype='float32')
            sd.wait()  # Wait until recording is finished

            print("Processing audio...")
            # Flatten the audio data
            audio_data = audio_data.flatten()

            # Preprocess the audio
            preprocessed_audio = preprocess_audio(audio_data, target_sample_rate)

            # Perform inference
            prediction = perform_inference(model, preprocessed_audio)
            print("Prediction:", prediction)

            print("Processing complete. Waiting for next cycle...\n")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        sys.exit(0)

# Run the main function
if __name__ == "__main__":
    main()
