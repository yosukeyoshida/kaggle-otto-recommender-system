lint:
	flake8 --show-source .

fmt:
	isort ./src/*.py
	black .
