PROJECT_ROOT  ?= $(CURDIR)
APP_DIR       := $(PROJECT_ROOT)/mushroom_id_app
JAVA_DIR      := $(PROJECT_ROOT)/java-backend
FLUTTER_BIN   ?= flutter
PYTHON_PORT   ?= 8000
JAVA_PORT     ?= 8080
WEB_PORT      ?= 8081
VENV_DIR      ?= $(PROJECT_ROOT)/.venv
VENV_PYTHON   := $(VENV_DIR)/bin/python
VENV_UVICORN  := $(VENV_DIR)/bin/uvicorn
OLLAMA_PORT    ?= 11434
OLLAMA_MODEL   ?= gemma3:4b
OLLAMA_BASE_URL ?= http://localhost:$(OLLAMA_PORT)
OLLAMA_TIMEOUT ?= 300
BENCHMARK_MANIFEST ?= $(PROJECT_ROOT)/benchmarks/evaluation_manifest.csv
BENCHMARK_OUTPUT_DIR ?= $(PROJECT_ROOT)/artifacts/benchmarks/local
BENCHMARK_METHODS ?= cnn tree db

# ---------------------------------------------------------------------------
# All targets declared phony
# ---------------------------------------------------------------------------
.PHONY: help \
        api java-backend \
        start stop \
        benchmark benchmark-local benchmark-unified benchmark-all benchmark-validate _benchmark-run \
        web-build web-serve web \
        flutter-analyze flutter-test \
        java-build java-run java-test \
        ollama-setup ollama \
        clean

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "Mushroom ID — available targets"
	@echo "────────────────────────────────────────────────────────────"
	@echo "  make api              Start Python FastAPI backend  (port $(PYTHON_PORT))"
	@echo "  make java-backend     Build + run Java Spring Boot  (port $(JAVA_PORT))"
	@echo "  make start            Start both backends in background"
	@echo "  make stop             Kill both backends"
	@echo ""
	@echo "  make web-build        Build Flutter web app"
	@echo "  make web-serve        Serve pre-built Flutter web  (port $(WEB_PORT))"
	@echo "  make web              Build and serve Flutter web"
	@echo ""
	@echo "  make flutter-analyze  Run dart analyze on Flutter app"
	@echo "  make flutter-test     Run Flutter unit tests"
	@echo ""
	@echo "  make java-build       Build Java JAR (requires Maven)"
	@echo "  make java-run         Run the Java JAR directly"
	@echo "  make java-test        Run Java tests (requires Maven)"
	@echo ""
	@echo "  make ollama-setup     Install Ollama + pull llama3.2:3b model"
	@echo "  make ollama           Start Ollama server in background"
	@echo ""
	@echo "  make benchmark        Run local benchmark without LLM (cnn tree db)"
	@echo "  make benchmark-local  Same as benchmark"
	@echo "  make benchmark-unified Run unified benchmark with Ollama/LLM"
	@echo "  make benchmark-all    Run all benchmark methods, including unified"
	@echo "  make benchmark-validate Check manifest and required local artifacts"
	@echo ""
	@echo "  make clean            Remove build artefacts"
	@echo "────────────────────────────────────────────────────────────"

# ---------------------------------------------------------------------------
# Python FastAPI backend  (Step 1-4 AI pipeline)
# ---------------------------------------------------------------------------
api:
	cd $(PROJECT_ROOT) && \
	$(VENV_UVICORN) api.main:app --reload --host 0.0.0.0 --port $(PYTHON_PORT)

# ---------------------------------------------------------------------------
# Java Spring Boot backend  (proxy + REST API for Flutter)
# ---------------------------------------------------------------------------
java-build:
	cd $(JAVA_DIR) && mvn -q clean package -DskipTests

java-run:
	java -jar $(JAVA_DIR)/target/mushroom-id-backend-*.jar \
	    --server.port=$(JAVA_PORT) \
	    --python.api.base-url=http://localhost:$(PYTHON_PORT)

java-backend: java-build
	java -jar $(JAVA_DIR)/target/mushroom-id-backend-*.jar \
	    --server.port=$(JAVA_PORT) \
	    --python.api.base-url=http://localhost:$(PYTHON_PORT)

java-test:
	cd $(JAVA_DIR) && mvn test

# ---------------------------------------------------------------------------
# Start / stop both backends together
# ---------------------------------------------------------------------------
start:
	@if [ ! -f $(JAVA_DIR)/target/mushroom-id-backend-*.jar ]; then \
	    echo "JAR not found — building first…"; \
	    cd $(JAVA_DIR) && mvn -q clean package -DskipTests; \
	fi
	@echo "Starting Python FastAPI on port $(PYTHON_PORT)…"
	cd $(PROJECT_ROOT) && \
	$(VENV_UVICORN) api.main:app --host 0.0.0.0 --port $(PYTHON_PORT) &
	@echo "Starting Java backend on port $(JAVA_PORT)…"
	java -jar $(JAVA_DIR)/target/mushroom-id-backend-*.jar \
	    --server.port=$(JAVA_PORT) \
	    --python.api.base-url=http://localhost:$(PYTHON_PORT) &
	@echo "Both backends started. Run 'make stop' to shut them down."

stop:
	@echo "Stopping backends…"
	@-lsof -ti :$(PYTHON_PORT) 2>/dev/null | xargs -r kill -9 || true
	@-lsof -ti :$(JAVA_PORT)   2>/dev/null | xargs -r kill -9 || true
	@-lsof -ti :$(WEB_PORT)    2>/dev/null | xargs -r kill -9 || true
	@echo "Backends stopped."

# ---------------------------------------------------------------------------
# Ollama local LLM
# ---------------------------------------------------------------------------
ollama-setup:
	@echo "Installing Ollama…"
	curl -fsSL https://ollama.com/install.sh | sh
	@echo "Pulling llama3.2:3b model (approx 2 GB)…"
	ollama pull llama3.2:3b
	@echo "Ollama setup complete. Run 'make ollama' to start the server."

ollama:
	@echo "Starting Ollama server on port $(OLLAMA_PORT)…"
	OLLAMA_PORT=$(OLLAMA_PORT) ollama serve &
	@echo "Ollama running. Restart the Python API ('make stop && make start') to activate LLM."

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
benchmark: benchmark-local

benchmark-local: BENCHMARK_METHODS := cnn tree db
benchmark-local: BENCHMARK_OUTPUT_DIR := $(PROJECT_ROOT)/artifacts/benchmarks/local
benchmark-local: _benchmark-run

benchmark-unified: BENCHMARK_METHODS := unified
benchmark-unified: BENCHMARK_OUTPUT_DIR := $(PROJECT_ROOT)/artifacts/benchmarks/unified
benchmark-unified: _benchmark-run

benchmark-all: BENCHMARK_METHODS := all
benchmark-all: BENCHMARK_OUTPUT_DIR := $(PROJECT_ROOT)/artifacts/benchmarks/comparative
benchmark-all: _benchmark-run

benchmark-validate:
	@test -x "$(VENV_PYTHON)" || { echo "Missing Python venv: $(VENV_PYTHON)"; exit 1; }
	@test -f "$(BENCHMARK_MANIFEST)" || { echo "Missing benchmark manifest: $(BENCHMARK_MANIFEST)"; exit 1; }
	@test -f "$(PROJECT_ROOT)/artifacts/cnn_weights.pt" || { echo "Missing CNN weights: $(PROJECT_ROOT)/artifacts/cnn_weights.pt"; exit 1; }
	@test -f "$(PROJECT_ROOT)/data/Yolov8/best.pt" || { echo "Missing YOLO weights: $(PROJECT_ROOT)/data/Yolov8/best.pt"; exit 1; }
	cd $(PROJECT_ROOT) && \
	$(VENV_PYTHON) -c "import csv; from pathlib import Path; manifest=Path('$(BENCHMARK_MANIFEST)'); rows=list(csv.DictReader(open(manifest, encoding='utf-8'))); species={r['species_id'] for r in rows}; missing=[str((manifest.parent / r[p]).resolve()) for r in rows for p in ('above_image_path','below_image_path') if r.get(p) and not (manifest.parent / r[p]).resolve().exists()]; bad=[(r['specimen_id'], r.get('confusing_pair_with')) for r in rows if r.get('confusing_pair_with') and r['confusing_pair_with']!='NONE' and r['confusing_pair_with'] not in species]; print(f'manifest_rows={len(rows)} species_count={len(species)} missing_paths={len(missing)} bad_confusing_pairs={len(bad)}'); raise SystemExit(1 if missing or bad else 0)"

_benchmark-run: benchmark-validate
	@mkdir -p "$(BENCHMARK_OUTPUT_DIR)"
	@echo "Running benchmark methods: $(BENCHMARK_METHODS)"
	@echo "Manifest: $(BENCHMARK_MANIFEST)"
	@echo "Output:   $(BENCHMARK_OUTPUT_DIR)"
	cd $(PROJECT_ROOT) && \
	OLLAMA_BASE_URL="$(OLLAMA_BASE_URL)" \
	OLLAMA_MODEL="$(OLLAMA_MODEL)" \
	OLLAMA_TIMEOUT="$(OLLAMA_TIMEOUT)" \
	$(VENV_PYTHON) -m benchmarks.run_comparative \
	    --manifest "$(BENCHMARK_MANIFEST)" \
	    --output-dir "$(BENCHMARK_OUTPUT_DIR)" \
	    --methods $(BENCHMARK_METHODS)

# ---------------------------------------------------------------------------
# Flutter web
# ---------------------------------------------------------------------------
web-build:
	cd $(APP_DIR) && \
	$(FLUTTER_BIN) clean && \
	$(FLUTTER_BIN) pub get && \
	$(FLUTTER_BIN) build web --no-wasm-dry-run

web-serve:
	cd $(APP_DIR)/build/web && \
	python3 -m http.server $(WEB_PORT)

web: web-build web-serve

# ---------------------------------------------------------------------------
# Flutter checks
# ---------------------------------------------------------------------------
flutter-analyze:
	cd $(APP_DIR) && \
	$(FLUTTER_BIN) analyze lib/

flutter-test:
	cd $(APP_DIR) && \
	$(FLUTTER_BIN) test

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	@echo "Stopping any running backends…"
	@-lsof -ti :$(PYTHON_PORT) 2>/dev/null | xargs -r kill -9 || true
	@-lsof -ti :$(JAVA_PORT)   2>/dev/null | xargs -r kill -9 || true
	@-lsof -ti :$(WEB_PORT)    2>/dev/null | xargs -r kill -9 || true
	@-lsof -ti :$(OLLAMA_PORT)          2>/dev/null | xargs -r kill -9 || true
	@-pgrep -f "ollama serve" 2>/dev/null | xargs -r kill -9 || true
	cd $(APP_DIR) && \
	$(FLUTTER_BIN) clean || true
	cd $(JAVA_DIR) && mvn -q clean || true
	@echo "Build artefacts cleaned."
