import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import librosa
import time
import io
import os
import urllib.request
import tarfile
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime

class YAMNetClassifier:
    """A comprehensive class to manage YAMNet audio classification."""

    def __init__(self, sample_rate: int = 16000, duration: int = 1):
        """
        Initialize the YAMNet classifier with configurable parameters.

        Args:
            sample_rate (int): Audio sampling rate. Defaults to 16000 Hz.
            duration (int): Duration of audio chunk to process. Defaults to 1 second.
        """
        self.SAMPLE_RATE = sample_rate
        self.DURATION = duration
        self.CHANNELS = 1
        self.MODEL_PATH = "yamnet_model"
        self.MODEL_URL = "https://storage.googleapis.com/tfhub-modules/google/yamnet/1.tar.gz"

        self.model = None
        self.class_names = None

    def download_yamnet_model(self) -> bool:
        """
        Download and extract the YAMNet model with a progress bar.

        Returns:
            bool: True if download and extraction is successful, False otherwise.
        """
        os.makedirs(self.MODEL_PATH, exist_ok=True)
        tar_path = os.path.join(self.MODEL_PATH, "yamnet_model.tar.gz")

        try:
            st.info(f"Downloading YAMNet model from {self.MODEL_URL}")

            with urllib.request.urlopen(self.MODEL_URL) as response, open(tar_path, 'wb') as out_file:
                total_size = int(response.headers.get('Content-Length', 0))
                block_size = 8192
                with tqdm(total=total_size, unit='i', unit_scale=True, desc="Downloading") as t:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        out_file.write(buffer)
                        t.update(len(buffer))

            st.info("Extracting YAMNet model...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=self.MODEL_PATH)

            os.remove(tar_path)
            st.success("YAMNet model downloaded and extracted successfully.")
            return True
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return False

    def load_model(self) -> Tuple[Optional[object], Optional[List[str]]]:
        """
        Load the YAMNet model and its class names.

        Returns:
            Tuple of (model, class_names) or (None, None) if loading fails.
        """
        if not os.path.exists(self.MODEL_PATH):
            success = self.download_yamnet_model()
            if not success:
                st.error("Failed to download YAMNet model.")
                return None, None

        try:
            model = hub.load(self.MODEL_PATH)
            labels_path = hub.resolve(os.path.join(self.MODEL_PATH, "assets", "yamnet_class_map.csv"))
            class_names = list(np.loadtxt(labels_path, delimiter=',', dtype=str, skiprows=1, usecols=[2]).flatten())
            return model, class_names
        except Exception as e:
            st.error(f"Error loading YAMNet model: {e}")
            return None, None

    def classify_audio(self, model, class_names, audio_data) -> List[Tuple[str, float]]:
        """
        Classify audio data using the YAMNet model.

        Args:
            model: Loaded YAMNet model
            class_names: List of sound class names
            audio_data: Audio data to classify

        Returns:
            List of top 5 classifications with their scores
        """
        try:
            results = model(audio_data)
            scores = results[0].numpy()
            mean_scores = np.mean(scores, axis=0)
            top_indices = np.argsort(mean_scores)[::-1][:5]
            self._create_spectrogram(audio_data)
            return [(class_names[i], mean_scores[i]) for i in top_indices]
        except Exception as e:
            st.error(f"Classification error: {e}")
            return []

    def record_and_classify(self, model, class_names, selected_device_index):
        """
        Record audio from microphone and classify in real-time.

        Args:
            model: Loaded YAMNet model
            class_names: List of sound class names
            selected_device_index: Index of the input audio device
        """
        try:
            with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=self.CHANNELS,
                                 dtype='float32', blocksize=int(self.SAMPLE_RATE * self.DURATION),
                                 device=selected_device_index) as stream:
                st.info(f"Recording from microphone: {sd.query_devices(selected_device_index)['name']}")
                main_placeholder = st.empty()
                sidebar_placeholder = st.sidebar.empty()
                final_log_placeholder = st.empty()

                if 'high_confidence_log' not in st.session_state:
                    st.session_state['high_confidence_log'] = {}
                if 'recording' not in st.session_state:
                    st.session_state['recording'] = True
                if 'stop_recording' not in st.session_state:
                    st.session_state['stop_recording'] = False
                if 'save_log' not in st.session_state:
                    st.session_state['save_log'] = False

                stop_button_pressed = st.button("🛑 Stop Recording", disabled=st.session_state.get('stop_recording', False))
                if stop_button_pressed:
                    st.session_state['stop_recording'] = True
                    st.session_state['recording'] = False

                start_time = time.time()

                while st.session_state['recording']:
                    audio_chunk, overflowed = stream.read(int(self.SAMPLE_RATE * self.DURATION))
                    if overflowed:
                        st.warning("Audio buffer overflowed!")

                    audio_chunk = audio_chunk.squeeze()
                    if audio_chunk.shape[0] == self.SAMPLE_RATE:
                        predictions = self.classify_audio(model, class_names, audio_chunk)
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        with main_placeholder.container():
                            st.subheader("Real-time Classification:")
                            if predictions:
                                for label, score in predictions:
                                    st.write(f"- {label}: {score:.2f}")
                            else:
                                st.write("No classification.")

                        for label, score in predictions:
                            if score > 0.6:
                                if label not in st.session_state['high_confidence_log']:
                                    st.session_state['high_confidence_log'][label] = {
                                        'first_occurred': current_time,
                                        'occurrences': 1,
                                        'total_confidence': score,
                                        'last_occurred': current_time
                                    }
                                else:
                                    log_entry = st.session_state['high_confidence_log'][label]
                                    log_entry['occurrences'] += 1
                                    log_entry['total_confidence'] += score
                                    log_entry['last_occurred'] = current_time
                                    st.session_state['high_confidence_log'][label] = log_entry

                        with sidebar_placeholder.container():
                            st.sidebar.subheader("High Confidence Classes (>60%):")
                            if st.session_state['high_confidence_log']:
                                data = []
                                for label, log_data in st.session_state['high_confidence_log'].items():
                                    avg_confidence = log_data['total_confidence'] / log_data['occurrences']
                                    data.append({
                                        "Class": label,
                                        "Occurrences": log_data['occurrences'],
                                        "Average Confidence": f"{avg_confidence * 100:.2f}%",
                                    })
                                df = pd.DataFrame(data)
                                st.sidebar.dataframe(df)
                            else:
                                st.sidebar.info("No high confidence classes detected yet.")

                    time.sleep(0.1)

                st.info("Real-time classification stopped.")

                if st.session_state['stop_recording'] and st.session_state['high_confidence_log']:
                    final_data = []
                    for label, log_data in st.session_state['high_confidence_log'].items():
                        avg_confidence = log_data['total_confidence'] / log_data['occurrences']
                        final_data.append({
                            "Time First Occurred": log_data['first_occurred'],
                            "Class": label,
                            "Occurrences": log_data['occurrences'],
                            "Average Confidence": f"{avg_confidence * 100:.2f}%",
                            "Last Occurred": log_data['last_occurred']
                        })
                    final_df = pd.DataFrame(final_data)

                    with final_log_placeholder.container():
                        st.subheader("Final Classification Log:")
                        st.dataframe(final_df)
                        if st.button("💾 Save Log to CSV"):
                            self._save_logs_to_csv(final_df, final=True)
                            st.session_state['high_confidence_log'] = {} # Clear log after saving
                            st.session_state['stop_recording'] = False # Reset stop state
                            st.rerun() # Rerun to clear the final table
                elif st.session_state['stop_recording']:
                    with final_log_placeholder.container():
                        st.info("No high confidence classes detected during the session.")
                    st.session_state['stop_recording'] = False # Reset stop state

        except sd.PortAudioError as e:
            st.error(f"Error with microphone input: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    def _create_spectrogram(self, audio_data: np.ndarray):
        """Generate waveform and spectrogram visualization."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        # Waveform
        ax1.plot(np.linspace(0, self.DURATION, len(audio_data)), audio_data)
        ax1.set_title('Audio Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')

        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=self.SAMPLE_RATE, ax=ax2)
        plt.colorbar(img, ax=ax2, format="%+2.0f dB")
        ax2.set_title('Spectrogram')

        st.pyplot(fig)

    def classify_audio_file(self, model, class_names, audio_file):
        try:
            audio_bytes = audio_file.read()
            with st.spinner("Processing audio file..."):
                audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=self.SAMPLE_RATE, mono=True)
                predictions = self.classify_audio(model, class_names, audio_array)

                st.subheader("Classification Result:")
                if predictions:
                    # Create a DataFrame for better visualization
                    results_df = pd.DataFrame(predictions, columns=["Sound Class", "Confidence"])
                    results_df["Confidence"] = results_df["Confidence"].apply(lambda x: f"{x * 100:.1f}%")

                    # Display the DataFrame
                    st.dataframe(results_df)

                    # Add a download button for the results
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="classification_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No classifications found for the audio file.")
        except Exception as e:
            st.error(f"Error processing audio file: {e}")

    def _save_logs_to_csv(self, df: pd.DataFrame, final: bool = False):
        """Saves the high confidence class logs to a CSV file."""
        file_path = r"D:\College_work\Projects\ExcelSoft\audio_classify\Google_Audio_Set\logs.csv"
        try:
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if final:
                if os.path.exists(file_path):
                    existing_df = pd.read_csv(file_path)
                    # Ensure all columns are present in both dataframes before concatenation
                    if 'Time First Occurred' not in existing_df.columns:
                        existing_df['Time First Occurred'] = None
                    if 'Last Occurred' not in existing_df.columns:
                        existing_df['Last Occurred'] = None
                    combined_df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(subset=['Time First Occurred', 'Class', 'Occurrences', 'Average Confidence', 'Last Occurred'], keep='last')
                    combined_df.to_csv(file_path, index=False)
                else:
                    df.to_csv(file_path, index=False)
                st.sidebar.success(f"Saved classification logs to {file_path}")
            else:
                df.to_csv(file_path, index=False, mode='a', header=not os.path.exists(file_path))
        except Exception as e:
            st.sidebar.error(f"Error saving logs to CSV: {e}")

def main():
    st.set_page_config(
        page_title="Audio Classification V3",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎧 Advanced Audio Classification with YAMNet V3")
    st.markdown("*Powered by TensorFlow Hub and YAMNet*")

    # Initialize the classifier
    classifier = YAMNetClassifier()

    # Load the model at the start of the app
    model, class_names = classifier.load_model()

    # Check if model and class names are loaded successfully
    if model is None or class_names is None:
        st.error("Failed to load YAMNet model. Please check your internet connection and try again.")
        return

    # Initialize session state for high confidence classes if not present
    if 'high_confidence_log' not in st.session_state:
        st.session_state['high_confidence_log'] = {}
    if 'recording' not in st.session_state:
        st.session_state['recording'] = False
    if 'stop_recording' not in st.session_state:
        st.session_state['stop_recording'] = False
    if 'save_log' not in st.session_state:
        st.session_state['save_log'] = False

    # Add a sidebar with model information
    st.sidebar.header("🔊 Model Information")
    st.sidebar.info(f"Total Sound Classes: {len(class_names)}")

    input_option = st.radio("Choose input source:", ("Microphone", "Audio File"))

    if input_option == "Microphone":
        try:
            devices = sd.query_devices()
            input_devices = [dev for dev in devices if dev['max_input_channels'] > 0]
            if not input_devices:
                st.warning("No input microphones found.")
                return

            device_names = [f"{dev['name']} (Index: {i})" for i, dev in enumerate(input_devices)]
            selected_device_name = st.selectbox("Select input microphone:", device_names)
            selected_device_index = int(selected_device_name.split('(Index: ')[1][:-1])

            st.info("⏱️ Note: Click 'Start' to begin real-time classification. Click 'Stop' to end and see the full log.")
            if st.button("🎙️ Start Real-time Classification") and not st.session_state['recording']:
                st.session_state['recording'] = True
                st.session_state['high_confidence_log'] = {} # Clear log on new recording
                st.session_state['stop_recording'] = False
                st.session_state['save_log'] = False
                st.rerun() # Force rerun to start the recording loop

            if st.session_state['recording']:
                classifier.record_and_classify(model, class_names, selected_device_index)
            elif st.session_state['stop_recording'] and st.button("💾 Save Log to CSV"):
                final_data = []
                for label, log_data in st.session_state['high_confidence_log'].items():
                    avg_confidence = log_data['total_confidence'] / log_data['occurrences']
                    final_data.append({
                        "Time First Occurred": log_data['first_occurred'],
                        "Class": label,
                        "Occurrences": log_data['occurrences'],
                        "Average Confidence": f"{avg_confidence * 100:.2f}%",
                        "Last Occurred": log_data['last_occurred']
                    })
                final_df = pd.DataFrame(final_data)
                classifier._save_logs_to_csv(final_df, final=True)
                st.session_state['high_confidence_log'] = {} # Clear log after saving
                st.session_state['stop_recording'] = False # Reset stop state
                st.rerun() # Rerun to clear the final table

        except sd.PortAudioError as e:
            st.error(f"Error initializing audio devices: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    elif input_option == "Audio File":
        st.info("🎵 Supports .wav and .mp3 files")
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

        if audio_file is not None:
            st.audio(audio_file, format='audio/wav')  # Preview the uploaded audio
            if st.button("🔍 Classify Audio File"):
                classifier.classify_audio_file(model, class_names, audio_file)

if __name__ == "__main__":
    main()