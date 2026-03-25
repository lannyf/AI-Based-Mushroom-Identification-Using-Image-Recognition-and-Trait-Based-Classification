# AI-Based Mushroom Identification Using Image Recognition and Trait-Based Classification

## 🚀 Quick Start - Run the Web App

From the root of this repository, run:

```bash
cd mushroom_id_app && flutter build web && cd build/web && python3 -m http.server 8080
```

Then open your browser to:
```
http://localhost:8080
```

✅ The app will load with full Material Design 3 interface  
⏹️ To stop: Press **Ctrl + C** in terminal

> **Tip:** You can also run the app directly in Chrome without building first:
> ```bash
> cd mushroom_id_app && flutter run -d chrome --web-port 8080
> ```

---

## 📚 Project Overview

This is a comprehensive AI-based mushroom identification system built with Flutter, featuring:
- **Multi-stage identification**: Image recognition + Trait classification + LLM analysis
- **Web & Mobile**: Runs on browsers (web) and Android devices
- **Smart safety warnings**: Detects lookalikes and toxic species
- **Full history tracking**: Stores all identifications locally
- **40+ automated tests**: Comprehensive test coverage across 4 components
- **Phase 6 Complete**: 100% Flutter mobile app ready

---

## 📋 Supported Image Formats

- **PNG** (recommended)
- **JPG/JPEG**
- Maximum file size: 5 MB

Example: Use `Mushroom_examples/svamp.png` for testing

---

## 🗂️ Project Structure

```
├── mushroom_id_app/           # Flutter web & mobile app
│   ├── lib/
│   │   ├── main.dart          # App entry point
│   │   ├── pages/             # 5 UI pages (camera, questionnaire, results, history, settings)
│   │   ├── providers/         # GetX state management
│   │   └── services/          # Business logic (image, storage)
│   ├── test/                  # 40+ automated tests
│   └── pubspec.yaml           # Dependencies
├── models/                    # Python ML models (Phase 1-5)
├── scripts/                   # Processing & utility scripts
├── Docs/                      # Project documentation
└── README.md                  # This file
```

---

## 🎯 Project Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1-5 | Backend System (Python) | ✅ 100% Complete |
| 6 | Flutter Mobile App | ✅ 100% Complete |
| 7 | Evaluation & Analysis | 🔄 In Progress |
| 8 | Documentation & Thesis | 📋 Planning |

**Overall**: 87.5% Complete (7 of 8 phases)

---

## 📱 Features

### Working Features
- ✅ Home page with navigation
- ✅ Questionnaire page (42 mushroom traits across 6 categories)
- ✅ Results page (AI predictions with confidence scores)
- ✅ Identification history (browser/device storage)
- ✅ App settings and preferences
- ✅ Material Design 3 UI with responsive design
- ✅ GetX reactive state management
- ✅ 40+ automated tests (85%+ coverage)

### Limited on Web
- ⚠️ Camera access (use Android APK for real camera)
- ⚠️ Direct device file system access

---

## 🧪 Testing

Run automated tests:
```bash
cd mushroom_id_app
flutter test
```

Expected output: 40+ tests passing with 85%+ coverage

---

## 📖 Documentation

- **EMULATOR-SETUP.md** - Android emulator setup instructions
- **DEPLOYMENT.md** - Build and release procedures
- **TESTING-GUIDE.md** - Comprehensive test documentation
- **implementationplan.md** - Detailed project implementation roadmap

---

## 🛠️ Technology Stack

**Frontend (Flutter)**
- Dart 3.11.3
- Flutter 3.41.5
- GetX state management
- Material Design 3

**Backend (Python)**
- TensorFlow/Keras ML models
- Trait-based classification
- LLM integration

**Data Storage**
- SQLite (local history)
- Browser storage (web)

---

## 🔗 Key Files

- `mushroom_id_app/lib/main.dart` - App configuration and theming
- `mushroom_id_app/lib/services/image_service.dart` - Image validation
- `mushroom_id_app/lib/providers/identification_provider.dart` - State management
- `mushroom_id_app/lib/pages/results_page.dart` - Results display with confidence scores

---

## 📝 License & Credits

Academic project for mushroom identification research.

---
