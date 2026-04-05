PROJECT_ROOT := /home/iannyf/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification
APP_DIR := $(PROJECT_ROOT)/mushroom_id_app
FLUTTER_BIN := $(HOME)/Emi/flutter_linux_3.41.5-stable/flutter/bin
PORT ?= 8080

.PHONY: web-build web-serve web serv api

web-build:
	cd $(APP_DIR) && \
	export PATH="$(FLUTTER_BIN):$$PATH" && \
	flutter clean && \
	flutter pub get && \
	flutter build web --no-wasm-dry-run

web-serve:
	cd $(APP_DIR)/build/web && \
	python3 -m http.server $(PORT)

web: web-build web-serve

api:
	cd $(PROJECT_ROOT) && \
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
