# RL Grid World — Makefile
# ========================
# Run these commands from your terminal:
#   make install    — Install Python dependencies
#   make train      — Train the agent (main command)
#   make test-env   — Test the grid world environment
#   make test-agent — Test the actor-critic agent
#   make test-viz   — Test the visualizations with fake data
#   make clean      — Remove generated files

PYTHON = python3

# Install dependencies
install:
	$(PYTHON) -m pip install -r requirements.txt

# Train the agent (the main event!)
train:
	$(PYTHON) train.py

# Test individual components
test-env:
	$(PYTHON) environment.py

test-agent:
	$(PYTHON) agent.py

test-viz:
	$(PYTHON) visualize.py

# Run all tests
test: test-env test-agent

# Remove generated files (plots, caches)
clean:
	rm -f training_rewards.png agent_path.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

.PHONY: install train test-env test-agent test-viz test clean
