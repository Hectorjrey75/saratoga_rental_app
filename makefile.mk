.PHONY: help install train predict app test clean

help:
	@echo "Available commands:"
	@echo "  make install    Install dependencies"
	@echo "  make train      Train the model"
	@echo "  make app        Run Streamlit app"
	@echo "  make test       Run tests"
	@echo "  make clean      Clean temporary files"
	@echo "  make lint       Run linters"
	@echo "  make format     Format code"

install:
	pip install -r requirements.txt
	pip install -e .

train:
	python -m src.models.model_training

app:
	streamlit run app/main.py

test:
	pytest tests/ -v --cov=src --cov-report=html

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

lint:
	flake8 src/ app/ tests/
	mypy src/ --ignore-missing-imports

format:
	isort src/ app/ tests/
	black src/ app/ tests/