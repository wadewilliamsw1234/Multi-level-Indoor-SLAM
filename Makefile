# ============================================
# NUFR-M3F SLAM Benchmarking Suite
# Makefile
# ============================================

.PHONY: all build build-base build-lego-loam build-orb-slam3 build-vins-fusion \
        build-basalt build-droid-slam build-evaluation \
        run run-lego-loam run-orb-slam3 run-vins-fusion run-basalt run-droid-slam \
        evaluate figures stats clean clean-metrics clean-results \
        test shell check-data help

# ============================================
# Configuration
# ============================================
DATA_DIR ?= ./data/ISEC
RESULTS_DIR ?= ./results
CONFIG_DIR ?= ./config

# Docker image names
IMAGE_BASE = slam-benchmark/ros-base
IMAGE_LEGO = slam-benchmark/lego-loam
IMAGE_ORB = slam-benchmark/orb-slam3
IMAGE_VINS = slam-benchmark/vins-fusion
IMAGE_BASALT = slam-benchmark/basalt
IMAGE_DROID = slam-benchmark/droid-slam
IMAGE_EVAL = slam-benchmark/evaluation

# ============================================
# Default target
# ============================================
all: help

# ============================================
# Help
# ============================================
help:
	@echo "NUFR-M3F SLAM Benchmarking Suite"
	@echo "================================"
	@echo ""
	@echo "Build Commands:"
	@echo "  make build              Build all Docker images"
	@echo "  make build-lego-loam    Build LeGO-LOAM image"
	@echo "  make build-orb-slam3    Build ORB-SLAM3 image"
	@echo "  make build-vins-fusion  Build VINS-Fusion image"
	@echo "  make build-evaluation   Build evaluation image"
	@echo ""
	@echo "Run Commands:"
	@echo "  make run                Run complete benchmark pipeline"
	@echo "  make run-lego-loam      Run LeGO-LOAM on all floors"
	@echo "  make run-orb-slam3      Run ORB-SLAM3 on all floors"
	@echo ""
	@echo "Analysis Commands:"
	@echo "  make evaluate           Run full evaluation pipeline"
	@echo "  make stats              Show quick trajectory statistics"
	@echo "  make figures            Generate all figures"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make test               Run unit tests"
	@echo "  make check-data         Verify dataset is accessible"
	@echo "  make shell              Open shell in ROS container"
	@echo "  make clean              Clean all generated results"
	@echo "  make clean-metrics      Clean metrics only (keep trajectories)"
	@echo ""
	@echo "Dataset should be at: $(DATA_DIR)"

# ============================================
# Build Targets
# ============================================
build: build-base build-lego-loam build-orb-slam3 build-evaluation
	@echo "All images built successfully"

build-base:
	docker build -t $(IMAGE_BASE):latest -f docker/Dockerfile.ros-base .

build-lego-loam: build-base
	docker build -t $(IMAGE_LEGO):latest -f docker/Dockerfile.lego-loam .

build-orb-slam3:
	docker build -t $(IMAGE_ORB):latest -f docker/Dockerfile.orb-slam3 .

build-vins-fusion: build-base
	docker build -t $(IMAGE_VINS):latest -f docker/Dockerfile.vins-fusion .

build-basalt:
	docker build -t $(IMAGE_BASALT):latest -f docker/Dockerfile.basalt .

build-droid-slam:
	docker build -t $(IMAGE_DROID):latest -f docker/Dockerfile.droid-slam .

build-evaluation:
	docker build -t $(IMAGE_EVAL):latest -f docker/Dockerfile.evaluation .

# ============================================
# Run Targets
# ============================================
run:
	docker compose up slam-benchmark

run-lego-loam:
	@echo "Running LeGO-LOAM on all floors..."
	@for floor in 5th_floor 1st_floor 4th_floor 2nd_floor; do \
		echo "Processing $$floor..."; \
		docker run --rm -it \
			-v $(DATA_DIR):/data/ISEC:ro \
			-v $(RESULTS_DIR):/results \
			--network=host \
			$(IMAGE_LEGO):latest \
			/root/run_lego_loam_floor.sh $$floor; \
	done
	@echo "LeGO-LOAM complete. Results in $(RESULTS_DIR)/trajectories/lego_loam/"

run-orb-slam3:
	@echo "Running ORB-SLAM3 on all floors..."
	docker run --rm -it \
		-v $(DATA_DIR):/data/ISEC:ro \
		-v $(RESULTS_DIR):/results \
		-v $(CONFIG_DIR):/config:ro \
		--network=host \
		$(IMAGE_ORB):latest \
		/root/run_orb_slam3_all.sh
	@echo "ORB-SLAM3 complete. Results in $(RESULTS_DIR)/trajectories/orb_slam3/"

run-orb-slam3-floor:
	@echo "Running ORB-SLAM3 on $(FLOOR)..."
	docker run --rm -it \
		-v $(DATA_DIR):/data/ISEC:ro \
		-v $(RESULTS_DIR):/results \
		-v $(CONFIG_DIR):/config:ro \
		--network=host \
		$(IMAGE_ORB):latest \
		/root/run_orb_slam3_floor.sh $(FLOOR)

run-vins-fusion:
	docker compose run vins-fusion

run-basalt:
	docker compose run basalt

run-droid-slam:
	docker compose run --gpus all droid-slam

# ============================================
# Analysis Targets
# ============================================
evaluate:
	@echo "Running evaluation..."
	python3 scripts/evaluation/evaluate_all.py
	@echo "Results saved to $(RESULTS_DIR)/metrics/"

stats:
	@echo "=== LeGO-LOAM Statistics ==="
	@python3 scripts/evaluation/quick_traj_stats.py $(RESULTS_DIR)/trajectories/lego_loam/
	@echo ""
	@echo "=== ORB-SLAM3 Statistics ==="
	@python3 scripts/evaluation/quick_traj_stats.py $(RESULTS_DIR)/trajectories/orb_slam3/ 2>/dev/null || echo "No ORB-SLAM3 results yet"

figures:
	@echo "Generating figures..."
	@mkdir -p $(RESULTS_DIR)/figures/lego_loam
	@mkdir -p $(RESULTS_DIR)/figures/orb_slam3
	@mkdir -p $(RESULTS_DIR)/figures/comparisons
	@for floor in 5th_floor 1st_floor 4th_floor 2nd_floor; do \
		python3 scripts/visualization/plot_trajectory_2d.py \
			$(RESULTS_DIR)/trajectories/lego_loam/$$floor.txt \
			$(RESULTS_DIR)/figures/lego_loam/$$floor.png 2>/dev/null || true; \
		python3 scripts/visualization/plot_trajectory_2d.py \
			$(RESULTS_DIR)/trajectories/orb_slam3/$$floor.txt \
			$(RESULTS_DIR)/figures/orb_slam3/$$floor.png 2>/dev/null || true; \
	done
	@echo "Figures saved to $(RESULTS_DIR)/figures/"

# ============================================
# Utility Targets
# ============================================
test:
	pytest tests/ -v

check-data:
	@echo "Checking dataset at $(DATA_DIR)..."
	@if [ -d "$(DATA_DIR)/5th_floor" ]; then \
		echo "✓ 5th_floor found"; \
		ls -1 $(DATA_DIR)/5th_floor/*.bag 2>/dev/null | wc -l | xargs echo "  Bags:"; \
	else \
		echo "✗ 5th_floor NOT found"; \
	fi
	@if [ -d "$(DATA_DIR)/1st_floor" ]; then \
		echo "✓ 1st_floor found"; \
	else \
		echo "✗ 1st_floor NOT found"; \
	fi
	@if [ -d "$(DATA_DIR)/4th_floor" ]; then \
		echo "✓ 4th_floor found"; \
	else \
		echo "✗ 4th_floor NOT found"; \
	fi
	@if [ -d "$(DATA_DIR)/2nd_floor" ]; then \
		echo "✓ 2nd_floor found"; \
	else \
		echo "✗ 2nd_floor NOT found"; \
	fi

shell:
	docker run --rm -it \
		-v $(DATA_DIR):/data/ISEC:ro \
		-v $(RESULTS_DIR):/results \
		-v $(CONFIG_DIR):/config:ro \
		--network=host \
		$(IMAGE_BASE):latest \
		/bin/bash

shell-lego:
	docker run --rm -it \
		-v $(DATA_DIR):/data/ISEC:ro \
		-v $(RESULTS_DIR):/results \
		--network=host \
		$(IMAGE_LEGO):latest \
		/bin/bash

shell-orb:
	docker run --rm -it \
		-v $(DATA_DIR):/data/ISEC:ro \
		-v $(RESULTS_DIR):/results \
		-v $(CONFIG_DIR):/config:ro \
		--network=host \
		$(IMAGE_ORB):latest \
		/bin/bash

# ============================================
# Clean Targets
# ============================================
clean-metrics:
	rm -rf $(RESULTS_DIR)/metrics/*.zip
	rm -rf $(RESULTS_DIR)/metrics/*.json
	rm -rf $(RESULTS_DIR)/figures/*
	rm -rf $(RESULTS_DIR)/logs/*
	@echo "Metrics and figures cleaned (trajectories preserved)"

clean-results:
	rm -rf $(RESULTS_DIR)/trajectories/*
	rm -rf $(RESULTS_DIR)/metrics/*
	rm -rf $(RESULTS_DIR)/figures/*
	rm -rf $(RESULTS_DIR)/logs/*
	rm -rf $(RESULTS_DIR)/*.bag
	@echo "All results cleaned"

clean: clean-results
	docker compose down -v
	@echo "Full cleanup complete"

# ============================================
# Docker Compose Targets
# ============================================
up:
	docker compose up

down:
	docker compose down

logs:
	docker compose logs -f
