#   make install    — Install Python dependencies
#   make train      — Train the agent (main command)
#   make test-env   — Test the grid world environment
#   make test-agent — Test the actor-critic agent
#   make test-viz   — Test the visualizations with fake data
#   make clean      — Remove generated files

PYTHON = python3

# Install dependencies

# Train the agent (the main event!)
train:
	$(PYTHON) train_cartpole.py

# Test individual components

test-agent:
	$(PYTHON) cartpole_agent.py

# Remove generated files (plots, caches)

.PHONY: install train test-env test-agent test-viz test clean
