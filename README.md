# Recognition and Classification of Traditional Mauritanian Music Notes Using LSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20Analysis-green)](https://librosa.org/)
[![LSTM](https://img.shields.io/badge/Deep%20Learning-LSTM-red)](https://github.com/MulayeMuhammad)

## 🎵 Project Overview

This project develops a **Deep Learning LSTM model** to automatically recognize and classify **Maqamat** (musical modes) in traditional Mauritanian music, specifically focusing on **Tidinit** (traditional lute) recordings. This innovative approach aims to preserve and analyze the rich cultural heritage of Mauritanian music using cutting-edge machine learning techniques.

**Biyaadh Louden** (بياض لوذن) - symbolizing the profound passion Mauritanians have for their traditional music - inspires this project to safeguard this unique musical heritage.

---

## 🎯 Objectives

1. **Automate Maqam Classification**: Develop an LSTM model capable of accurately classifying Mauritanian musical recordings by their Maqam
2. **Preserve Cultural Heritage**: Provide a tool for analyzing and preserving traditional Mauritanian music
3. **Audio Feature Extraction**: Extract relevant audio features (MFCCs, harmonics, zero-crossing rate) using Librosa
4. **Deep Learning Application**: Apply LSTM architecture to capture temporal dependencies in musical sequences
5. **Support Musicological Research**: Facilitate the study and understanding of Mauritanian musical traditions

---

## 🎼 About Mauritanian Traditional Music

### The Tidinit (التِدْنِيتْ)

The **Tidinit** is the emblematic stringed instrument at the heart of Mauritanian music, played primarily by male griots. It is a four-stringed lute that produces the melodic foundation of traditional compositions.

### The Ardin (الأَرْدِينْ)

The **Ardin** is a traditional West African harp with 9-14 strings, constructed from a calabash, and played by female griots (griottes) in Mauritanian music.

### Maqamat of AZAWAN

In Mauritanian traditional music, **Maqam** (plural: Maqamat) refers to a musical mode used to structure and organize melodies. Each Maqam has its own characteristics and is used to evoke specific emotions.

The five main Maqamat of **AZAWAN** (أزوان) are:

1. **Karr** (كَرّ) - Represents strength and power
2. **Vaghou** (ڤَغُو) - Evokes contemplation and spirituality  
3. **Lkhall** (لْخَلّ) - Associated with lightness and joy
4. **Lbyad** (لْبْيَضْ) - Symbolizes purity and serenity
5. **Lebteyt** (لْبْتِيتْ) - Conveys melancholy and nostalgia

Each Maqam is characterized by specific melodic patterns, intervals, and emotional qualities that are deeply rooted in Mauritanian cultural tradition.

---

## 🔬 Methodology

### 1. Audio Feature Extraction

Using **Librosa** library, we extract comprehensive audio features:

```python
- MFCCs (Mel-Frequency Cepstral Coefficients)
- Zero-Crossing Rate
- Harmonics and Percussive components
- Spectral features (centroid, bandwidth, contrast)
- Chroma features
- Tempo and beat tracking
```

### 2. Data Preprocessing

```python
- Load audio files (.wav, .mp3)
- Normalize audio signals
- Segment into fixed-size windows
- Pad/truncate sequences for uniform input
- Split into train/validation/test sets (70/15/15)
```

### 3. LSTM Model Architecture

```
Input Layer (Audio Features)
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
LSTM Layer 2 (64 units)
    ↓
Dropout (0.3)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Output Layer (5 classes - Softmax)
```

### 4. Training Process

```python
- Loss function: Categorical Crossentropy
- Optimizer: Adam (learning rate=0.001)
- Epochs: 100 with early stopping
- Batch size: 32
- Callbacks: ModelCheckpoint, ReduceLROnPlateau
```

---

## 📁 Project Structure

```
Recognition-Traditional-Mauritanian-Music-LSTM/
│
├── data/
│   ├── audio_files/              # Tidinit recordings (*.wav, *.mp3)
│   │   ├── karr/
│   │   ├── vaghou/
│   │   ├── lkhall/
│   │   ├── lbyad/
│   │   └── lebteyt/
│   └── features/                 # Extracted features (*.npy)
│
├── notebooks/
│   └── rim_trad_music_recognizer.ipynb    # Main training notebook
│
├── models/
│   ├── lstm_model.h5            # Trained LSTM model
│   ├── model_architecture.json  # Model architecture
│   └── scaler.pkl               # Feature scaler
│
├── src/
│   ├── feature_extraction.py    # Librosa feature extraction
│   ├── preprocessing.py         # Data preprocessing utilities
│   ├── model.py                 # LSTM model definition
│   ├── train.py                 # Training pipeline
│   └── predict.py               # Inference script
│
├── results/
│   ├── confusion_matrix.png     # Classification results
│   ├── training_history.png     # Loss/accuracy curves
│   └── maqam_predictions.csv    # Prediction results
│
├── presentation/
│   └── Projet_Deep_Learning-4.pdf    # Project presentation
│
├── requirements.txt
├── README.md                    # This file
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installation

```bash
# Clone the repository
git clone https://github.com/MulayeMuhammad/Recognition-and-Classification-of-Traditional-Mauritanian-Music-Notes-Using-LSTM.git
cd Recognition-and-Classification-of-Traditional-Mauritanian-Music-Notes-Using-LSTM

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
tensorflow>=2.10.0
librosa>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
soundfile>=0.11.0
```

---

## 💻 Usage

### 1. Feature Extraction

```python
from src.feature_extraction import extract_features

# Extract features from audio file
features = extract_features('path/to/audio.wav')
```

### 2. Training the Model

```python
# Run training script
python src/train.py --epochs 100 --batch_size 32

# Or use the Jupyter notebook
jupyter notebook notebooks/rim_trad_music_recognizer.ipynb
```

### 3. Making Predictions

```python
from src.predict import predict_maqam

# Predict Maqam for new audio
maqam = predict_maqam('path/to/new_audio.wav')
print(f"Predicted Maqam: {maqam}")
```

---

## 📊 Dataset

### Audio Recordings

The dataset consists of traditional Mauritanian music recordings featuring **Tidinit** and **Ardin** instruments, labeled by their corresponding Maqam.

**⚠️ Important: Data Access**

Due to the cultural sensitivity and copyright considerations of traditional Mauritanian music recordings, the audio dataset is **not included** in this repository.

**To access the dataset for research or educational purposes:**

📧 **Contact us**: mulayemuhammad@gmail.com

Please provide:
- Your name and affiliation
- Purpose of use (research, education, cultural preservation)
- Brief description of your project

We are committed to preserving and promoting Mauritanian musical heritage while respecting intellectual property and cultural rights.

### Dataset Statistics

- **Total recordings**: [To be specified after data collection]
- **Maqamat classes**: 5 (Karr, Vaghou, Lkhall, Lbyad, Lebteyt)
- **Duration range**: 10-60 seconds per sample
- **Sample rate**: 22,050 Hz
- **Format**: WAV, MP3

---

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Training Accuracy** | [To be filled] |
| **Validation Accuracy** | [To be filled] |
| **Test Accuracy** | [To be filled] |
| **Precision** | [To be filled] |
| **Recall** | [To be filled] |
| **F1-Score** | [To be filled] |

### Confusion Matrix

[Insert confusion matrix visualization here after training]

### Per-Maqam Performance

| Maqam | Precision | Recall | F1-Score |
|-------|-----------|---------|----------|
| Karr | [TBF] | [TBF] | [TBF] |
| Vaghou | [TBF] | [TBF] | [TBF] |
| Lkhall | [TBF] | [TBF] | [TBF] |
| Lbyad | [TBF] | [TBF] | [TBF] |
| Lebteyt | [TBF] | [TBF] | [TBF] |

---

## 🎓 Technical Details

### Audio Processing Pipeline

```python
1. Load audio file (Librosa)
2. Resample to 22,050 Hz
3. Extract MFCCs (13 coefficients)
4. Compute delta and delta-delta MFCCs
5. Extract zero-crossing rate
6. Separate harmonics and percussives
7. Normalize features (StandardScaler)
8. Create sequences for LSTM input
```

### LSTM Hyperparameters

```python
{
    "lstm_units_1": 128,
    "lstm_units_2": 64,
    "dropout_rate": 0.3,
    "dense_units": 32,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "sequence_length": 100
}
```

---

## 🌍 Cultural Significance

### Biyaadh Louden (بياض لوذن)

The term **"Biyaadh Louden"** represents the intense appreciation and passion that Mauritanians feel for their traditional music. This love for music is so strong that they are ready to defend and preserve these melodies with total devotion. Biyaadh Louden illustrates this sacred relationship between Mauritanians and their musical heritage.

This project honors this cultural treasure by:
- Providing tools for preservation and analysis
- Facilitating musicological research
- Promoting Mauritanian cultural heritage globally
- Supporting education about traditional music theory

---

## 🔮 Future Work

- [ ] Expand dataset with more recordings from different regions
- [ ] Include other Mauritanian instruments (Tbal, Tinidit variations)
- [ ] Develop real-time Maqam recognition application
- [ ] Create mobile app for music enthusiasts and students
- [ ] Implement attention mechanisms for improved performance
- [ ] Multi-instrument classification
- [ ] Temporal segmentation of Maqam transitions
- [ ] Integration with music notation software
- [ ] Web-based interactive demo

---

## 👥 Team

**Project developed by:**

- **Ahmed Bezeid Ahmed Abd El Aziz** (ID: 22068)
- **Moulaye Ahmed Mohamed BRAHIM** (ID: 22281)
- **Zeinebou Taki** (ID: 22092)

**Institution**: École Supérieure Polytechnique (ESP), Mauritania

**Course**: SID47 - Machine Learning (2024)

---

## 👨‍💻 Main Author & Contact

**Moulaye Ahmed Mohammed Brahim**

- 🌐 Portfolio: [mulayemuhammad.github.io/Moulaye_DS_Portfolio](https://mulayemuhammad.github.io/Moulaye_DS_Portfolio/)
- 💼 LinkedIn: [Moulaye Ahmed MUHAMMAD](https://www.linkedin.com/in/moulaye-ahmed-muhammad/)
- 🐙 GitHub: [@MulayeMuhammad](https://github.com/MulayeMuhammad)
- 📧 Email: mulayemuhammad@gmail.com
- 🐦 Twitter: [@MuhammadMoulaye](https://twitter.com/MuhammadMoulaye)

**For dataset access or collaboration inquiries**: mulayemuhammad@gmail.com

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note on Cultural Content**: While the code is open-source, the musical recordings represent Mauritanian cultural heritage and may be subject to additional usage restrictions. Please contact us for clarification.

---

## 🙏 Acknowledgments

- **École Supérieure Polytechnique (ESP)** for academic support and resources
- **Mauritanian griots and musicians** for preserving this rich musical tradition
- **Traditional music experts** who provided insights into Maqam theory
- **Librosa team** for excellent audio analysis tools
- **TensorFlow/Keras team** for deep learning framework
- All who contribute to preserving and promoting Mauritanian cultural heritage

---

## 📚 References

### Traditional Mauritanian Music
1. Guignard, M. (1975). "Musique, honneur et plaisir au Sahara." *Librairie Orientaliste Paul Geuthner*.
2. Norris, H. T. (1968). "Shinqiti Folk Literature and Song." *Clarendon Press*.
3. Touma, H. H. (1996). "The Music of the Arabs." *Amadeus Press*.

### Technical References
4. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.
5. McFee, B., et al. (2015). "librosa: Audio and Music Signal Analysis in Python." *SciPy*.
6. Humphrey, E. J., Bello, J. P., & LeCun, Y. (2012). "Moving beyond feature design: Deep architectures and automatic feature learning in music informatics." *ISMIR*.

---

## 🎵 About AZAWAN

**AZAWAN** (أزوان) is the traditional Mauritanian musical repertoire that encompasses the five Maqamat. It represents centuries of oral tradition passed down through generations of griots, preserving the cultural memory and artistic expression of Mauritanian society.

---

## 🌟 Project Impact

This project contributes to:

- ✅ **Cultural Preservation**: Digital archiving of musical traditions
- ✅ **Education**: Teaching tools for music students
- ✅ **Research**: Enabling musicological studies
- ✅ **Technology Transfer**: Applying ML to cultural heritage
- ✅ **Global Awareness**: Promoting Mauritanian music internationally

---

## 🔊 Audio Examples

[Links to sample audio clips will be added - Contact for access]

---

## 📊 Project Status

- [x] Literature review and methodology design
- [x] Feature extraction pipeline implementation
- [x] LSTM model architecture design
- [ ] Complete dataset collection (In progress)
- [ ] Model training and optimization
- [ ] Performance evaluation
- [ ] Web application development
- [ ] Documentation and publication

---

<p align="center">
  <img src="https://via.placeholder.com/600x200/8B4513/FFFFFF?text=AZAWAN+-+5+Maqamat" alt="AZAWAN Maqamat">
</p>

<p align="center">
  <i>⭐ If you appreciate Mauritanian traditional music or are interested in cultural AI applications, please star this project!</i>
</p>

<p align="center">
  <strong>Preserving Cultural Heritage Through AI 🎵🤖</strong>
</p>

<p align="center">
  Made with ❤️ and respect for Mauritanian musical tradition by <a href="https://github.com/MulayeMuhammad">Moulaye Ahmed</a>
</p>

---

<p align="center">
  <sub>بياض لوذن - The Passion for Music Lives On</sub>
</p>
