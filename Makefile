all:
	python -m srcs.trainAndPredict.main

train:
	python -m srcs.trainAndPredict.main 10 10 train

predict:
	python -m srcs.trainAndPredict.main 10 10 predict

vizualize:
	python -m srcs.vizualisation.main
