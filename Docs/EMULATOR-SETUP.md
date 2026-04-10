# Smartphone Simulator Setup Guide for Flutter Testing

## Current Situation

The Flutter SDK is not currently installed on this machine. To test the Mushroom Identification app, we need to set up:

1. **Flutter SDK** (if not installed)
2. **Android SDK & Emulator** (for Android testing)
3. **Android Virtual Device (AVD)** (Android smartphone simulator)

---

## Prerequisites Check

### Operating System
- **Current:** Linux (likely Ubuntu/Debian)
- **Supported:** Linux, macOS, Windows

### Required Tools
- Java Development Kit (JDK) 8+
- Android SDK
- Android Emulator

---

## Installation Steps

### Step 1: Install Flutter SDK

#### Option A: Download from flutter.dev
```bash
# Navigate to home directory
cd ~

# Download Flutter SDK
git clone https://github.com/flutter/flutter.git -b stable

# Add Flutter to PATH
export PATH="$PATH:~/flutter/bin"

# Add to ~/.bashrc or ~/.zshrc for permanent effect
echo 'export PATH="$PATH:~/flutter/bin"' >> ~/.bashrc
source ~/.bashrc
```

#### Option B: Using package manager
```bash
sudo apt-get update
sudo apt-get install flutter
```

### Step 2: Verify Flutter Installation

```bash
flutter --version
flutter doctor  # Shows all dependencies and issues
```

The `flutter doctor` output will show:
- ✓ Flutter (Channel stable)
- ✓ Android toolchain
- ✓ Android SDK
- ✓ Android emulator
- ✓ Dart SDK

### Step 3: Accept Android SDK Licenses

```bash
flutter doctor --android-licenses

# Type 'y' and press Enter for each prompt
```

### Step 4: Create Android Virtual Device (AVD)

#### Using Command Line
```bash
# Create a Pixel 5 emulator with Android 13
avdmanager create avd -n mushroom-pixel5 \
  -k "system-images;android-33;google_apis;x86_64" \
  -d "Pixel 5"

# Verify creation
avdmanager list avd
```

#### Using Android Studio (GUI)
1. Open Android Studio
2. Tools → Device Manager
3. Create device → Select Pixel 5
4. Select Android 13 (API 33)
5. Finish

### Step 5: Start the Android Emulator

```bash
# Start in background with good performance
emulator -avd mushroom-pixel5 \
  -cores 4 \
  -memory 2048 \
  -partition-size 512 \
  -gpu auto &
```

**Wait 30-60 seconds for full boot**

### Step 6: Verify Emulator Connection

```bash
# After emulator boots, check connection
adb devices

# Expected output:
# List of attached devices
# emulator-5554          device
```

### Step 7: Run Flutter App on Emulator

```bash
# Navigate to app directory
cd /home/iannyf/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification/mushroom_id_app

# Run on emulator
flutter run

# Or specify device explicitly
flutter run -d emulator-5554
```

---

## Quick Start Script

Save this as `run_app.sh`:

```bash
#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Starting mushroom ID emulator...${NC}"

# Start emulator
emulator -avd mushroom-pixel5 \
  -cores 4 \
  -memory 2048 \
  -gpu auto &

EMULATOR_PID=$!

# Wait for boot
echo -e "${YELLOW}Waiting 40 seconds for emulator to boot...${NC}"
sleep 40

# Check connection
echo -e "${YELLOW}Waiting for device to be ready...${NC}"
adb wait-for-device

# Navigate and run
cd /home/iannyf/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification/mushroom_id_app

echo -e "${GREEN}Launching Flutter app...${NC}"
flutter run

# Cleanup on exit
echo -e "${YELLOW}Shutting down emulator...${NC}"
kill $EMULATOR_PID
```

Run with: `chmod +x run_app.sh && ./run_app.sh`

---

## Recommended Configurations

### For Fast Development
```bash
emulator -avd mushroom-pixel5 \
  -cores 4 \
  -memory 2048 \
  -gpu auto \
  -no-audio \
  -no-snapshot-load &
```

### For Realistic Testing
```bash
emulator -avd mushroom-pixel5 \
  -cores 2 \
  -memory 1024 \
  -gpu swiftshader \
  -no-audio &
```

### For Headless (CI/CD)
```bash
emulator -avd mushroom-pixel5 \
  -no-window \
  -no-audio \
  -cores 2 \
  -memory 2048 &
```

---

## Testing the App

### Run All Tests
```bash
cd mushroom_id_app
flutter test
```

### Run on Emulator
```bash
# Release build (optimized)
flutter run --release

# Profile build (performance monitoring)
flutter run --profile

# Debug build (with logging)
flutter run
```

### View Logs
```bash
flutter logs          # Live logs
flutter logs > app.log  # Save to file
```

---

## Manual Testing Workflow

### 1. Start Emulator & App
```bash
# Terminal 1: Start emulator
emulator -avd mushroom-pixel5 -cores 4 -memory 2048 -gpu auto &

# Terminal 2: Run app (after 30 seconds)
cd mushroom_id_app && flutter run
```

### 2. Test Camera Page
- Tap "Take Photo" button
- Device will prompt for camera permission
- Take a photo or select from gallery
- Rotate/zoom/pan the image
- Tap "Confirm" to proceed

### 3. Test Questionnaire Page
- You'll see 6 pages with different traits
- Select options using radio buttons
- Progress bar shows current page
- Swipe or use Previous/Next buttons
- Tap "Submit" to validate and proceed

### 4. Test Results Page
- View confidence percentage (should be ~85%)
- See method breakdown (Image/Trait/LLM)
- Check top 5 predictions
- View lookalike warnings
- See safety rating (edible/caution/inedible)

### 5. Test History Page
- Tap "View History" button on home
- After identification, results appear here
- Tap entry to view details
- Delete individual entries
- Clear all history

### 6. Test Settings Page
- Tap settings icon (top right on home)
- Toggle notifications on/off
- Change language (English, Swedish, German, French)
- Edit API base URL
- View privacy policy and terms

---

## Database Inspection

### View Data Saved in Emulator
```bash
# List app databases
adb shell ls /data/data/com.example.mushroom_identification/databases/

# Access database directly
adb shell sqlite3 /data/data/com.example.mushroom_identification/databases/mushroom_identification.db

# Query tables
sqlite> SELECT * FROM history;
sqlite> SELECT * FROM preferences;
sqlite> .tables
sqlite> .schema history
sqlite> .quit
```

### Pull Database for Analysis
```bash
# Copy database to computer
adb pull /data/data/com.example.mushroom_identification/databases/mushroom_identification.db

# Open with SQLite
sqlite3 mushroom_identification.db

# Or with GUI tool (DB Browser for SQLite)
```

---

## Performance Monitoring

### View Logs
```bash
flutter logs
flutter logs -c  # Clear screen between logs
```

### Monitor Memory
```bash
adb shell dumpsys meminfo com.example.mushroom_identification
```

### Monitor CPU/Frame Rate
```bash
flutter run  # Shows frame rates in terminal
# Look for "Frame time:" in output
```

### Profile App
```bash
flutter run --profile
# Uses Dart DevTools for profiling
```

---

## Troubleshooting

### Problem: Emulator Won't Start

```bash
# Solution 1: Clear and restart
emulator -avd mushroom-pixel5 -wipe-data

# Solution 2: Check port
netstat -tulpn | grep 5554

# Solution 3: Kill all emulators and restart
pkill -f emulator
sleep 2
emulator -avd mushroom-pixel5 &
```

### Problem: App Crashes on Launch

```bash
# Clear app data
adb shell pm clear com.example.mushroom_identification

# Reinstall fresh
flutter clean
flutter pub get
flutter run
```

### Problem: Device Not Found

```bash
# Restart ADB server
adb kill-server
adb start-server

# List devices
adb devices

# Wait for device
adb wait-for-device
```

### Problem: Slow Emulator

```bash
# Increase resources
emulator -avd mushroom-pixel5 \
  -cores 4 \
  -memory 2048 \
  -partition-size 512

# Or check system resources
free -h    # Available RAM
df -h      # Available disk space
```

### Problem: Camera Not Working

```bash
# Camera is available in emulator
# Grant permission when prompted
# Check logs for camera-related errors
flutter logs | grep -i camera
```

---

## Performance Benchmarks

Expected performance on typical machine:

| Operation | Time |
|-----------|------|
| Emulator startup | 30-60 seconds |
| App launch | 2-5 seconds |
| Page navigation | <300ms |
| Database operations | <100ms |
| Image processing | 1-3 seconds |
| Test suite run | 30-60 seconds |

---

## Complete Testing Checklist

- [ ] Flutter SDK installed
- [ ] Android SDK installed
- [ ] Android Virtual Device created
- [ ] Emulator starts successfully
- [ ] App launches on emulator
- [ ] Camera page works
- [ ] Questionnaire page navigates
- [ ] Results display correctly
- [ ] History saves entries
- [ ] Settings page functional
- [ ] All 70+ tests pass
- [ ] Database contains data
- [ ] No errors in logs

---

## Next Steps

1. **Install Flutter SDK** (`flutter doctor` first)
2. **Accept Android licenses** (`flutter doctor --android-licenses`)
3. **Create emulator** (`avdmanager create avd...`)
4. **Start emulator** (`emulator -avd mushroom-pixel5 ...`)
5. **Run app** (`flutter run` in mushroom_id_app)
6. **Test features** (camera, form, results, history, settings)
7. **Run tests** (`flutter test`)
8. **Inspect database** (`adb pull` and query)

---

## Useful Commands

```bash
# Check all dependencies
flutter doctor

# List devices
adb devices

# List emulators
emulator -list-avds

# Create new emulator
avdmanager create avd -n name -k "system-images;android-33;google_apis;x86_64" -d "Pixel 5"

# Start emulator
emulator -avd mushroom-pixel5 &

# Stop emulator
adb emu kill

# Run app
flutter run

# Run tests
flutter test

# View logs
flutter logs

# Build release
flutter build apk --release

# Clean and rebuild
flutter clean && flutter pub get && flutter run
```

---

**Once complete, you'll have a fully functional testing environment!** ✅
