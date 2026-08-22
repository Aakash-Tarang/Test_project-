# Top-level convenience targets. The C++ engine is built via CMake (src/).
.PHONY: env check build test report clean

env:
	./environment/setup.sh

check:
	./environment/check.sh

# Run the data pipeline (download->clean->QA->summary). SRC=github|yahoo
data:
	./scripts/data/run_pipeline.sh --source $(or $(SRC),github)

# Build the C++ engine (from Part 3 onward)
build:
	cmake -S src -B build
	cmake --build build -j2

test: build
	./build/statarbsim --test || true

# Build the report (canonical LaTeX + HTML preview)
report:
	python3 report/build_report.py

clean:
	rm -rf build
