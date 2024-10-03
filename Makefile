make install:
	poetry install
	poetry run pre-commit install

make test:
	poetry run pytest

make lint:
	poetry run pre-commit run --all-files

make run:
	poetry run uvicorn greencompute_backend.main:app --reload

make build:
	docker build -t greencompute-backend .

make docker-run:
	docker run -p 8000:8000 -d --name gc-backend --env-file .env greencompute-backend
