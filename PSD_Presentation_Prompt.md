# 📊 PSD — Predictive Sign Decoder
## PowerPoint Presentation Guide (10 Minutes)

> Use this as a slide-by-slide blueprint to build your PowerPoint. Each section = 1 slide (unless noted). Suggested time allocations are included.

---

## Slide 1 — Title Slide *(30 sec)*

- **Title:** PSD — Predictive Sign Decoder
- **Subtitle:** Real-Time Indian Sign Language to Text & Speech Communication System
- **Team Members:** Hetvi Pandav (+ other team members if any)
- **Department:** BE – Artificial Intelligence & Machine Learning
- **College Name & Logo**
- **Guide Name / Faculty Mentor**
- **Academic Year:** 2025–2026
- **Background Visual:** A clean image of hands making sign language gestures with digital overlays

---

## Slide 2 — Introduction *(45 sec)*

- **What is PSD?**
  - PSD (Predictive Sign Decoder) is a real-time Indian Sign Language (ISL) recognition system that converts hand gestures into readable English text and speech.
- **Who is it for?**
  - Hearing-impaired and mute individuals who use ISL to communicate
  - Bridges the communication gap between sign language users and non-signers
- **How does it work? (1-liner)**
  - Uses a webcam to detect hand gestures via Computer Vision & Machine Learning, then translates them into text, sentences, and speech — all in real time.

---

## Slide 3 — Problem Statement *(1 min)*

- **Communication Barrier:** ~6.3 million hearing-impaired people in India rely on Indian Sign Language, but the general public cannot understand it.
- **Limited ISL Resources:** Unlike ASL, Indian Sign Language has far fewer digital tools, datasets, and recognition systems available.
- **Dependency on Interpreters:** Deaf individuals depend on human sign language interpreters for daily communication, hospital visits, education, and government offices — interpreters are scarce and expensive.
- **No Real-Time Solution Exists:** Most existing systems are research prototypes that work offline or only on static images — none provide a usable, real-time, deployable desktop application.
- **Core Problem:** *"There is no accessible, real-time system that can convert Indian Sign Language gestures into readable text and speech for everyday use."*

---

## Slide 4 — Objectives *(45 sec)*

1. Build a real-time ISL recognition system using Computer Vision and Machine Learning
2. Recognize **static signs** (A–Z alphabet) using hand landmark features with a Random Forest classifier
3. Recognize **dynamic signs** (words like "Hello", "Thank You") using LSTM-based sequential models
4. Implement **intelligent motion-based switching** between static and dynamic sign modes
5. Develop a **smart sentence builder** that constructs readable sentences from detected signs
6. Provide **live translation** (English ↔ Hindi) and **Text-to-Speech** output
7. Deliver a **single-click deployable desktop application** (`.exe`) with a web-based UI
8. Make the system **accessible, lightweight, and usable** without technical knowledge

---

## Slide 5 — Literature Review / Existing Systems *(1 min)*

| # | System / Paper | Approach | Limitation |
|---|---------------|----------|------------|
| 1 | Google's MediaPipe Hands | Real-time 21-point hand tracking | Only tracking, no sign language recognition |
| 2 | ASL Fingerspelling (Various) | CNN on static hand images for American Sign Language | ASL only; no ISL support; static signs only |
| 3 | SignAll (Commercial) | Depth sensors + gloves for sign recognition | Requires expensive hardware (Kinect/gloves) |
| 4 | DeepSign (Research) | Deep learning on video sequences for word-level signs | Offline processing; not real-time; no sentence building |
| 5 | ISL Recognition (Academic papers) | SVM / KNN on landmark features | Low accuracy; static signs only; no deployment |

**Research Gap Identified:**
- No system combines **static + dynamic ISL** recognition in real-time
- No existing tool provides **sentence formation + translation + TTS** in one package
- No **deployable desktop application** for ISL exists

---

## Slide 6 — Proposed Methodology *(1.5 min, use 2 sub-slides or a flowchart)*

### System Architecture Flowchart:

```
Webcam Input
    ↓
MediaPipe Hand Detection (21 landmarks × 3 coords × 2 hands = 126 features)
    ↓
Motion Analysis (np.linalg.norm of keypoint difference)
    ↓
┌──────────────────────┬──────────────────────────┐
│  Low Motion (Static) │  High Motion (Dynamic)   │
│  ↓                   │  ↓                       │
│  Random Forest       │  LSTM Neural Network     │
│  Classifier          │  (30-frame sequences)    │
│  → Letter (A–Z)      │  → Word (Hello, Thank U) │
└──────────┬───────────┴───────────┬──────────────┘
           ↓                       ↓
      Smart Sentence Builder (auto spacing, capitalization)
           ↓
      Live Translation (English ↔ Hindi via deep-translator)
           ↓
      Text-to-Speech Output (gTTS)
           ↓
      Web UI (Flask + Socket.IO + Real-Time Updates)
```

### Key Design Decisions:
- **Motion-based switching** instead of separate mode buttons — makes interaction natural
- **126-feature vector** (both hands) instead of single hand — supports two-handed signs
- **Locking & cooldown logic** — prevents duplicate predictions for the same gesture
- **ONNX Runtime** for dynamic model inference — faster than native TensorFlow at runtime

---

## Slide 7 — Work Completed in Phase I *(1 min)*

| Task | Status |
|------|--------|
| Dataset collection for static signs (A–Z) using MediaPipe | ✅ Complete |
| Static model training (Random Forest, 300 estimators) | ✅ Complete |
| Dataset collection for dynamic signs (Hello, Thank You) as `.npy` sequences | ✅ Complete |
| Dynamic model training (LSTM, 30 epochs) | ✅ Complete |
| ONNX model conversion for optimized inference | ✅ Complete |
| Real-time combined demo (static + dynamic detection) | ✅ Complete |
| Motion-based auto-switching logic | ✅ Complete |
| Smart sentence builder with spacing, backspace, and clear | ✅ Complete |
| Flask web application with Socket.IO real-time updates | ✅ Complete |
| Translation support (English ↔ Hindi) | ✅ Complete |
| Text-to-Speech integration (gTTS) | ✅ Complete |
| PyInstaller packaging into `SignBridge.exe` | ✅ Complete |
| Frontend UI (live camera feed, sign display, sentence, translation) | ✅ Complete |

---

## Slide 8 — Tools and Technologies Used *(45 sec)*

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.x | Core development language |
| **Computer Vision** | OpenCV | Webcam capture & frame processing |
| **Hand Tracking** | MediaPipe Hands | 21-landmark detection per hand |
| **Static ML** | Scikit-learn (Random Forest) | Alphabet classification (A–Z) |
| **Dynamic DL** | TensorFlow / Keras (LSTM) | Word-level sequence recognition |
| **Inference** | ONNX Runtime | Optimized model serving |
| **Data** | NumPy, Pandas | Feature extraction & dataset handling |
| **Backend** | Flask + Flask-SocketIO | Web server with real-time push updates |
| **Frontend** | HTML / CSS / JavaScript | User interface |
| **Translation** | deep-translator (Google) | English ↔ Hindi live translation |
| **TTS** | gTTS (Google Text-to-Speech) | Speaking detected sentences |
| **Packaging** | PyInstaller | Single `.exe` deployment |
| **Version Control** | Git + Git LFS | Code & large file management |

---

## Slide 9 — Challenges Faced *(45 sec)*

1. **MediaPipe import issues with PyInstaller** — Resolved using resilient fallback imports and `sys._MEIPASS` resource pathing
2. **"Paging file too small" error** — Caused by Flask debug mode + memory-mapped model files; fixed by disabling debug mode and using `mmap_mode=None` in joblib
3. **Static sign flickering** — Multiple rapid predictions for the same gesture; resolved with frame-count locking and cooldown logic
4. **Motion threshold tuning** — Finding the right balance between static and dynamic detection sensitivity required extensive experimentation (tested values from 0.008 to 0.2)
5. **Dynamic model loading delay** — Solved via lazy loading (ONNX model loads only on first dynamic sign detection)
6. **Two-hand feature alignment** — Ensuring consistent 126-feature vector even when only one hand is visible (zero-padding the missing hand)
7. **Real-time UI updates** — Replaced polling with WebSocket (Socket.IO) for seamless frontend updates without page refresh

---

## Slide 10 — Future Work *(45 sec)*

1. **Expand dynamic vocabulary** — Add more ISL words and phrases beyond "Hello" and "Thank You"
2. **Sentence grammar correction** — Integrate NLP models (e.g., GPT-based) to refine raw letter/word sequences into grammatically correct sentences
3. **Two-way communication** — Add Speech-to-Sign conversion (text/voice → animated sign language avatar)
4. **Mobile deployment** — Port system to Android/iOS using TensorFlow Lite for on-device inference
5. **Multi-language translation** — Expand beyond Hindi to support Gujarati, Marathi, Tamil, and other Indian languages
6. **Cloud-based model updates** — Allow OTA model improvements as new signs are added to the dataset
7. **User customization** — Let users record and train their own custom signs
8. **Accessibility integration** — Partner with educational institutions and government accessibility programs for real-world deployment

---

## Slide 11 — Timeline / Gantt Chart *(30 sec)*

| Phase | Task | Timeline |
|-------|------|----------|
| **Phase I** | Literature review & problem definition | Month 1 |
| **Phase I** | Dataset collection (static signs A–Z) | Month 1–2 |
| **Phase I** | Static model training (Random Forest) | Month 2 |
| **Phase I** | Dataset collection (dynamic signs) | Month 2–3 |
| **Phase I** | Dynamic model training (LSTM) | Month 3 |
| **Phase I** | Real-time combined demo | Month 3–4 |
| **Phase I** | Sentence builder + Flask web app | Month 4 |
| **Phase I** | Translation, TTS, and UI polish | Month 4–5 |
| **Phase I** | PyInstaller packaging & testing | Month 5 |
| **Phase II** | Vocabulary expansion & NLP refinement | Month 6–7 |
| **Phase II** | Mobile app development | Month 7–8 |
| **Phase II** | Testing, documentation & final submission | Month 8–9 |

> 💡 **Tip:** Convert this table into an actual **Gantt chart visual** in PowerPoint using SmartArt or a bar chart for better visual impact.

---

## Slide 12 — Conclusion *(30 sec)*

- **PSD (Predictive Sign Decoder)** successfully demonstrates a working, real-time ISL recognition system
- Combines **static (A–Z) + dynamic (words)** sign recognition in a single unified pipeline
- Features **intelligent motion-based switching**, **smart sentence building**, **live translation**, and **text-to-speech**
- Delivered as a **single-click desktop application** — no installation required
- Addresses a real and critical accessibility need for India's hearing-impaired community
- **Phase I is fully functional** — ready for vocabulary expansion and mobile deployment in Phase II

> *"PSD bridges the communication gap between sign language users and the hearing world through real-time AI-powered gesture recognition."*

---

## Slide 13 — Live Demo / Thank You *(if time permits)*

- Show a **30-second live demo** or a **recorded video** of the system in action
- Display the web UI with:
  - Live camera feed
  - Real-time sign detection
  - Sentence building
  - Translation toggle
- **Thank You slide** with team contact information and GitHub repository link:
  `https://github.com/HetviPandav123/sign-language-smart-communication`

---

## 🎨 Design Tips for PowerPoint

| Tip | Details |
|-----|---------|
| **Theme** | Use a modern dark theme with teal/blue accent colors |
| **Fonts** | Use **Poppins** or **Montserrat** for headings, **Open Sans** for body |
| **Icons** | Use flat icons for each technology (Python, TensorFlow, Flask, etc.) |
| **Flowchart** | Use SmartArt or a custom diagram for the methodology slide |
| **Gantt Chart** | Use a horizontal bar chart with colored phases |
| **Screenshots** | Include actual UI screenshots from the running application |
| **Videos** | Embed short demo clips on the demo slide |
| **Animations** | Use subtle slide transitions (Morph or Fade) — avoid overdoing it |
