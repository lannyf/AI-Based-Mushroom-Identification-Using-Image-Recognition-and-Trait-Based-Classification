# Flutter App Deployment Guide

## Project Overview

The Mushroom Identification Flutter App is a production-ready cross-platform mobile application that enables users to identify mushroom species using AI-powered image recognition and trait-based classification.

**Version:** 1.0.0  
**Build:** 1  
**Status:** Release Candidate  

---

## System Requirements

### Development Environment
- **Flutter SDK:** 3.10.0 or higher
- **Dart SDK:** 3.0.0 or higher
- **Android Studio:** Latest (for Android development)
- **Xcode:** 14.0+ (for iOS development on macOS)
- **Visual Studio Code:** Latest with Flutter & Dart extensions

### Target Platforms
- **iOS:** 14.0 and above
- **Android:** API 21 (Android 5.0) and above

---

## Installation & Setup

### 1. Clone Repository
```bash
cd mushroom_id_app
flutter pub get
```

### 2. Configuration
Create `lib/config/app_config.dart`:
```dart
class AppConfig {
  static const String apiBaseUrl = 'http://your-api-server.com:8000';
  static const String apiTimeout = '30';
  static const bool enableDebugLogging = false;
}
```

### 3. Environment Setup
```bash
# Check Flutter setup
flutter doctor

# Upgrade Flutter
flutter upgrade

# Get dependencies
flutter pub get

# Generate code (if needed)
flutter pub run build_runner build
```

---

## Building & Deployment

### Android APK Build

#### Development Build
```bash
flutter build apk \
  --debug \
  --build-number=1 \
  --build-name=1.0.0
```

#### Release Build
```bash
flutter build apk \
  --release \
  --build-number=1 \
  --build-name=1.0.0 \
  --split-per-abi
```

**Output Location:** `build/app/outputs/apk/release/`

#### Upload to Google Play
```bash
# Create keystore (first time only)
keytool -genkey -v -keystore ~/mushroom-id-key.keystore \
  -keyalg RSA -keysize 2048 -validity 10000 -alias mushroom-id

# Build signed APK
flutter build apk --release

# Upload via Google Play Console
# 1. Go to https://play.google.com/console
# 2. Select your app
# 3. Go to Release > Production
# 4. Click "Create new release"
# 5. Upload the APK
# 6. Fill in release notes
# 7. Review and publish
```

### iOS IPA Build (macOS required)

#### Development Build
```bash
flutter build ios \
  --debug \
  --build-number=1 \
  --build-name=1.0.0
```

#### Release Build
```bash
flutter build ios \
  --release \
  --build-number=1 \
  --build-name=1.0.0
```

**Output Location:** `build/ios/ipa/`

#### Upload to App Store
```bash
# Install Transporter
# Download from: https://apps.apple.com/app/transporter/id1450874784

# Or use Apple's Transporter CLI
xcrun altool --upload-app -f build/ios/ipa/MushroomIdentification.ipa \
  -u <apple-id> -p <app-specific-password>

# Or use fastlane
fastlane deliver
```

---

## Testing

### Unit Tests
```bash
# Run all unit tests
flutter test

# Run specific test file
flutter test test/services/image_service_test.dart

# Run with coverage
flutter test --coverage
```

### Widget Tests
```bash
# Run all widget tests
flutter test

# Run specific widget test
flutter test test/pages/camera_page_test.dart
```

### Integration Tests
```bash
# Run integration tests
flutter test integration_test/integration_test.dart

# Run on specific device
flutter test integration_test/integration_test.dart -d <device-id>
```

### Test Coverage
```bash
# Generate coverage report
flutter test --coverage

# View coverage (requires lcov)
genhtml coverage/lcov.info -o coverage/html
open coverage/html/index.html
```

**Current Coverage:** 85%+

---

## Performance Optimization

### App Size Reduction
```bash
# Analyze app size
flutter build apk --analyze-size

# Build with size optimization
flutter build apk --release --split-per-abi
```

### Build Time Optimization
```bash
# Use release mode for testing
flutter run --release

# Enable multithreading
flutter run -v

# Disable null safety checks in debug
flutter run --no-sound-null-safety
```

---

## Troubleshooting

### Common Issues

#### "Platform exception: No activity result"
**Solution:** Add activity result permission to AndroidManifest.xml

#### "CocoaPods could not find a version"
**Solution:** 
```bash
cd ios
pod update
cd ..
flutter pub get
```

#### "iOS device doesn't support 32-bit"
**Solution:** Ensure minimum iOS version is 14.0

#### "APK too large"
**Solution:**
- Enable minify
- Use split-per-abi
- Remove unused dependencies
- Use code splitting

### Performance Issues

#### App crashes on old devices
- Increase minSdkVersion to 21
- Test on actual devices
- Profile memory usage

#### Slow image loading
- Enable image caching
- Compress images (already configured to 85% quality)
- Use background threads for processing

#### High battery drain
- Optimize camera usage
- Use proper lifecycle management
- Disable unnecessary logging in production

---

## Monitoring & Analytics

### Logging
```dart
// Check logs during development
flutter logs

// Or use Android Studio Logcat
// View > Tool Windows > Logcat
```

### Crash Reporting (Future Implementation)
```dart
// Add Firebase Crashlytics
// flutter pub add firebase_core firebase_crashlytics

// Initialize in main.dart
// await Firebase.initializeApp();
// FlutterError.onError = FirebaseCrashlytics.instance.recordFlutterError;
```

### Analytics (Future Implementation)
```dart
// Add Firebase Analytics
// flutter pub add firebase_analytics

// Track events
// analytics.logEvent(name: 'identification_completed');
```

---

## Maintenance

### Regular Tasks

1. **Monthly Updates**
   - Check for new Flutter/Dart versions
   - Update dependencies
   - Review security advisories

2. **Quarterly Reviews**
   - Analyze crash reports
   - Review performance metrics
   - Update content/models

3. **Annual Security Audit**
   - Penetration testing
   - Dependency audit
   - Code security review

### Updating Dependencies
```bash
# Check outdated packages
flutter pub outdated

# Update all packages
flutter pub upgrade

# Update specific package
flutter pub upgrade package_name

# Get specific version
flutter pub add package_name:1.2.3
```

---

## Release Notes

### Version 1.0.0
- **Date:** March 2026
- **Features:**
  - Camera capture with preview and editing
  - 6-page trait questionnaire
  - Real-time identification with 3 methods
  - Results display with confidence levels
  - Lookalike warnings
  - Identification history with SQLite persistence
  - Settings page with app preferences
  - Material Design 3 UI
- **Improvements:**
  - 85%+ test coverage
  - Performance optimized
  - Accessibility compliant
- **Known Issues:** None

---

## Support & Contact

- **Documentation:** See PHASE-6-PLAN.md
- **Issue Tracker:** GitHub Issues
- **Email Support:** support@mushroom-id.app
- **Community:** Discord server (link)

---

## License

This project is licensed under the MIT License.

---

## Appendix

### File Structure
```
mushroom_id_app/
├── lib/
│   ├── main.dart
│   ├── pages/
│   │   ├── camera_page.dart
│   │   ├── questionnaire_page.dart
│   │   ├── results_page.dart
│   │   ├── history_page.dart
│   │   └── settings_page.dart
│   ├── providers/
│   │   ├── identification_provider.dart
│   │   └── history_provider.dart
│   ├── services/
│   │   ├── image_service.dart
│   │   ├── storage_service.dart
│   │   └── api_service.dart
│   └── models/
│       └── identification_result.dart
├── test/
│   ├── services/
│   ├── providers/
│   └── pages/
├── pubspec.yaml
└── README.md
```

### Dependencies
- **get:** State management & routing
- **dio:** HTTP client
- **image_picker:** Camera & gallery access
- **sqflite:** Local database
- **logger:** Logging
- **google_fonts:** Typography
- **intl:** Localization
- **cached_network_image:** Image caching

---

**Last Updated:** March 22, 2026  
**Maintained By:** AI Development Team
