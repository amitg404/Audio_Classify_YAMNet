# Audio_Classify_YAMNet

## 🎧 Advanced Real-time Audio Classification System

AudioClassify is a comprehensive audio classification application that uses Google's YAMNet model to identify and categorize sounds in real-time. Perfect for acoustic monitoring, environmental sound analysis, and audio research applications.

![AudioClassify](https://github.com/yourusername/AudioClassify/raw/main/docs/images/screenshot.png)

## ✨ Features

- **Real-time Audio Classification**: Process live audio from your microphone with continuous updates
- **Audio File Analysis**: Upload and analyze .wav or .mp3 files
- **High-Confidence Detection**: Track and log sounds detected with high confidence (>60%)
- **Visualization**: View waveforms and spectrograms of analyzed audio
- **Comprehensive Logging**: Save classification results with timestamps and confidence scores
- **User-friendly Interface**: Clean, intuitive Streamlit interface

## 🔊 Sound Classification

The system uses Google's YAMNet model, which can recognize over 500 different audio classes including:
- Human sounds (speech, laughter, singing)
- Animal sounds (barking, meowing, chirping)
- Musical instruments
- Vehicle and machinery noises
- Environmental sounds
- And many more...

## 🛠️ Technologies Used

- **Streamlit**: Interactive web application framework
- **TensorFlow & TensorFlow Hub**: Machine learning framework and YAMNet model
- **Librosa**: Audio analysis and processing
- **SoundDevice**: Microphone input handling
- **Pandas & Matplotlib**: Data handling and visualization

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AudioClassify.git
cd AudioClassify
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 📋 Requirements

```
streamlit>=1.15.0
tensorflow>=2.8.0
tensorflow-hub>=0.12.0
numpy>=1.21.0
sounddevice>=0.4.4
librosa>=0.9.0
pandas>=1.3.0
matplotlib>=3.5.0
tqdm>=4.62.0
```

## 💻 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Choose your input source:
   - **Microphone**: For real-time classification
   - **Audio File**: For analyzing pre-recorded audio (.wav or .mp3 format)

3. For microphone input:
   - Select your microphone device from the dropdown
   - Click "Start Real-time Classification"
   - Watch as sounds are classified in real-time
   - Click "Stop Recording" when done
   - Review and save your classification log

4. For audio file input:
   - Upload your audio file
   - Click "Classify Audio File"
   - View the analysis results and spectrogram
   - Download results as CSV if desired

## 🧠 How It Works

AudioClassify uses Google's YAMNet, a pre-trained deep neural network that can identify 521 different audio events. The application:

1. Captures audio in real-time or loads it from a file
2. Processes the audio in chunks (default: 1-second segments)
3. Passes the processed audio through the YAMNet model
4. Displays top predictions with confidence scores
5. Generates audio visualizations including waveform and spectrogram
6. Logs high-confidence detections (>60%) with timestamps

The first time you run the application, it will automatically download the YAMNet model (approximately 17MB).

## 📊 Data Visualization

The system provides:
- Real-time prediction display with confidence scores
- Dynamic tracking of high-confidence sound classes
- Waveform visualization of audio segments
- Spectrogram visualization for frequency analysis
- CSV export for further analysis in other tools

## 🔧 Customization

You can customize the application by modifying:
- The confidence threshold (default: 60%)
- The audio sampling rate (default: 16kHz)
- The duration of audio chunks (default: 1 second)
- The log file path (currently set to "C:\\AudioClassify\\logs.csv")

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🙏 Acknowledgements

- Google's YAMNet model for audio classification
- The TensorFlow and TensorFlow Hub teams
- The Streamlit team for their incredible framework

---

Developed with ❤️ for audio analysis enthusiasts
