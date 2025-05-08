CODE = src/nlotrajectories tests

.PHONY: pretty lint

pretty:
	isort $(CODE)
	black $(CODE)

lint:
	black --check $(CODE)
	isort --diff $(CODE)
	flake8 --statistics $(CODE)
