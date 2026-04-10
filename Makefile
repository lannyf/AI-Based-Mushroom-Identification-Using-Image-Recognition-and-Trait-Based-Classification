PROJECT_ROOT  ?= $(CURDIR)
APP_DIR       := $(PROJECT_ROOT)/mushroom_id_app
JAVA_DIR      := $(PROJECT_ROOT)/java-backend
FLUTTER_BIN   ?= flutter
PYTHON_PORT   ?= 8000
JAVA_PORT     ?= 8080
WEB_PORT      ?= 8081
VENV_DIR      ?= /home/iannyf/mushroom-venv
VENV_UVICORN  := $(VENV_DIR)/bin/uvicorn

# ---------------------------------------------------------------------------
# All targets declared phony
# ---------------------------------------------------------------------------
.PHONY: help \
        api java-backend \
        start stop \
        web-build web-serve web \
        flutter-analyze flutter-test \
        java-build java-run java-test \
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
	cd $(APP_DIR) && \
	$(FLUTTER_BIN) clean || true
	cd $(JAVA_DIR) && mvn -q clean || true
	@echo "Build artefacts cleaned."
