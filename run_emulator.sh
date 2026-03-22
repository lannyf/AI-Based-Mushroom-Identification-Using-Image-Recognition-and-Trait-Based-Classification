#!/bin/bash

# Mushroom Identification App - Emulator Quick Start
# This script sets up and runs the Flutter app on Android emulator

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
AVD_NAME="mushroom-pixel5"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-"$SCRIPT_DIR/mushroom_id_app"}"

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Mushroom Identification - Emulator Setup & Runner      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print section
print_section() {
    echo -e "${YELLOW}▶ $1${NC}"
}

# Function to check command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}✗ $1 not found. Please install $1 first.${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ $1 found${NC}"
    return 0
}

# Function to wait for device
wait_for_device() {
    print_section "Waiting for device..."
    timeout=0
    while [ $timeout -lt 60 ]; do
        if adb devices | grep -q "device$"; then
            echo -e "${GREEN}✓ Device ready${NC}"
            return 0
        fi
        sleep 2
        timeout=$((timeout + 2))
    done
    echo -e "${RED}✗ Device not ready after 60 seconds${NC}"
    return 1
}

# Step 1: Check prerequisites
print_section "Checking prerequisites..."
check_command "flutter" || exit 1
check_command "adb" || exit 1
check_command "emulator" || exit 1

# Step 2: Check Flutter setup
print_section "Checking Flutter setup..."
echo "Flutter version:"
flutter --version | head -1

# Step 3: List available emulators
print_section "Available emulators:"
if emulator -list-avds | grep -q "$AVD_NAME"; then
    echo -e "${GREEN}✓ $AVD_NAME found${NC}"
else
    echo -e "${YELLOW}! $AVD_NAME not found. Creating...${NC}"
    echo "Run: avdmanager create avd -n $AVD_NAME -k 'system-images;android-33;google_apis;x86_64' -d 'Pixel 5'"
    exit 1
fi

# Step 4: Check if emulator is already running
print_section "Checking for running emulator..."
if adb devices | grep -q "emulator"; then
    echo -e "${YELLOW}! Emulator already running${NC}"
    USE_EXISTING=true
else
    echo "No emulator running. Will start one."
    USE_EXISTING=false
fi

# Step 5: Start emulator if needed
if [ "$USE_EXISTING" = false ]; then
    print_section "Starting emulator..."
    echo "Command: emulator -avd $AVD_NAME -cores 4 -memory 2048 -gpu auto &"
    emulator -avd "$AVD_NAME" \
        -cores 4 \
        -memory 2048 \
        -gpu auto \
        -no-audio \
        -no-snapshot-load &
    
    EMULATOR_PID=$!
    echo "Emulator PID: $EMULATOR_PID"
    
    echo -e "${YELLOW}Waiting 45 seconds for emulator to boot...${NC}"
    sleep 45
    
    wait_for_device || exit 1
fi

# Step 6: Check emulator status
print_section "Emulator status:"
adb devices

# Step 7: Navigate to app directory
print_section "Checking app directory..."
if [ ! -d "$APP_DIR" ]; then
    echo -e "${RED}✗ App directory not found: $APP_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✓ App directory found${NC}"

# Step 8: Run Flutter pub get
print_section "Getting Flutter dependencies..."
cd "$APP_DIR"
flutter pub get

# Step 9: Run app
print_section "Launching Flutter app on emulator..."
echo "Running: flutter run"
flutter run

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    App is running!                         ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║  Press 'q' in terminal to exit the app                     ║${NC}"
echo -e "${GREEN}║  Use 'flutter logs' in another terminal to view logs      ║${NC}"
echo -e "${GREEN}║  Run 'flutter test' to run automated tests                ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
