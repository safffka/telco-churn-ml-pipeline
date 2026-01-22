features:
	python -m src.data_pipeline

train:
	python -m src.train

evaluate:
	python -m src.evaluate

report:
	python -m src.report

all:
	make features
	make train
	make evaluate
	make report
test:
	PYTHONPATH=/app pytest -q

monitor:
	python -m src.monitor
