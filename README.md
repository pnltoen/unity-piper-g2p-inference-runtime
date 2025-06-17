# unity-piper-g2p-inference-runtime

[![Watch the demo](https://img.youtube.com/vi/tHMdiyMR3i8/0.jpg)](https://www.youtube.com/watch?v=tHMdiyMR3i8)

## 🎙️ Project Overview

This project demonstrates a **REALTIME**, fully **ON-DEVICE**, and **LIGHTWEIGHT** Text-to-Speech (TTS) system implemented in Unity.

Powered by the **Unity Inference Engine** and optimized through **asynchronous processing**, both **grapheme-to-phoneme (G2P)** conversion and **speech synthesis** run locally—without requiring any external APIs or internet connection.

Each model is highly compact:  
- **Low / Medium quality voices** are around **60MB**,  
- **High-quality voices** remain under **108MB**,  
making this solution ideal for mobile and embedded applications.

By integrating the **Piper TTS engine**, the system supports **multiple voices and languages**, delivering smooth and responsive audio generation even on mid-range hardware—completely offline.

---
## 🚀 Usage

1. Clone the repository:  
   git clone https://github.com/pnltoen/unity-piper-g2p-inference-runtime.git  
   cd unity-piper-g2p-inference-runtime  

2. Download [`StreamingAssets.zip`](https://drive.google.com/file/d/17rJyCFnemh5MqJvGjNWWyLT0kOdiRCmF/view?usp=sharing) and extract it into the `Assets/` folder.  
   Make sure the full path becomes: `Assets/StreamingAssets/`

3. Open the Unity project and load the scene located at:  
   `Assets/piper.unity`

4. Download the VisionOS-style UI asset from the [Unity Asset Store](https://assetstore.unity.com/packages/tools/gui/ui-kit-for-vision-pro-os-265406)

5. In the **Hierarchy**, select the `piper_engine` object and adjust the **Piper TTS scales**:  
   - `noise_scale` (Default: `0.4`, Trained: `0.667`)  
   - `length_scale` (Default: `1.1`, Trained: `1`)  
   - `noise_w` (Default: `0.6`, Trained: `0.8`)

6. Press **Play** in the Editor and type the sentence you want to synthesize.

7. You can select different voices from the left panel.  
   For additional voices, refer to [piper/VOICES.md](https://github.com/rhasspy/piper/blob/master/VOICES.md)


## 🔍 Limitations and Notes

- `mini-bart-g2p` operates on a **word-level basis**, so context-sensitive pronunciations (e.g., `read` in past vs. present tense, or `use` as noun vs. verb) are not accurately distinguished.

- While **Piper** supports multilingual voices, this project uses `mini-bart-g2p`, which only accepts **English text** as input. Non-English text will not be converted properly to phonemes.

- Each input sentence is processed **one word per frame**, where the G2P encoder-decoder runs per word.  
  After all words are processed, the phoneme sequence is passed asynchronously to the Piper model for **real-time audio synthesis and playback**.

- Although `mini-bart-g2p` can run on GPU, its **auto-regressive** structure requires downloading predicted tokens to the CPU at each step.  
  This frequent GPU-to-CPU transfer introduces overhead, making pure CPU execution faster and more efficient in this case.

- The **Piper TTS model** runs on the **GPU** to ensure smooth real-time speech synthesis.  
  The entire inference process—from phoneme input to audio playback—is handled **asynchronously** for low-latency performance.  
  You can fine-tune responsiveness by adjusting the `k_LayersPerFrame` variable, which controls how many decoder layers are processed per frame.

## 📚 Additional Resources

For more implementation details and practical usage of Unity Inference Engine with TTS and G2P, refer to the following (Korean blog posts):

- [Implementing Piper TTS using Unity Inference Engine](https://pnltoen.tistory.com/entry/Unity-Inference-Engine%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%9C-Piper-TTS-Text-To-Speech-%EA%B5%AC%ED%98%84) *(Korean blog post)*
- [Exploring Grapheme-to-Phoneme (G2P) in Unity Inference Engine](https://pnltoen.tistory.com/entry/unity-inference-engine%EC%97%90%EC%84%9C-g2p-grapheme-to-phoneme-%EA%B3%A0%EC%B0%B0) *(Korean blog post)*
- [Preprocessing with Python Server in Unity Inference Engine](https://pnltoen.tistory.com/entry/%EC%9C%A0%EB%8B%88%ED%8B%B0%EC%97%90%EC%84%9C-Python-%EC%84%9C%EB%B2%84-%ED%99%9C%EC%9A%A9%ED%95%B4-Unity-Inference-Engine-%EC%A0%84%EC%B2%98%EB%A6%AC-%ED%95%B4%EB%B3%B4%EA%B8%B0) *(Korean blog post)*

---
## 📄 License

This project (`unity-piper-g2p-inference-runtime`) is released under the **Apache License 2.0**.

However, it includes third-party machine learning models and phonemization tools that are governed by their own licenses:

### ▶️ Included Models and Licenses

- [mini-bart-g2p (Cisco AI)](https://huggingface.co/cisco-ai/mini-bart-g2p) – **Apache License 2.0**
- [Piper TTS (Rhasspy)](https://github.com/rhasspy/piper) – **MIT License**


All external models used in this project are serialized and stored in `Assets/StreamingAssets/`.  
Their original license texts are also included in that directory.

> ⚠️ Please review and comply with each license when redistributing or modifying any part of this project that includes third-party components.
