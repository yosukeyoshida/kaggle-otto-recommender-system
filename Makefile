lint:
	flake8 --show-source .

fmt:
	isort *.py
	black .
