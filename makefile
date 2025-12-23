PACKAGE_NAME = FLiESANN
ENVIRONMENT_NAME = $(PACKAGE_NAME)
DOCKER_IMAGE_NAME = $(shell echo $(PACKAGE_NAME) | tr '[:upper:]' '[:lower:]')

clean:
	rm -rf *.o *.out *.log
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

test:
	pytest -vv

build:
	python -m build

twine-upload:
	twine upload dist/*

dist:
	make clean
	make build
	make twine-upload

install:
	pip install -e .[dev]

uninstall:
	pip uninstall $(PACKAGE_NAME)

reinstall:
	make uninstall
	make install

environment:
	mamba create -y -n $(ENVIRONMENT_NAME) -c conda-forge python=3.11

remove-environment:
	mamba env remove -y -n $(ENVIRONMENT_NAME)

colima-start:
	colima start -m 16 -a x86_64 -d 100 

docker-build:
	docker build -t $(DOCKER_IMAGE_NAME):latest .

docker-build-environment:
	docker build --target environment -t $(DOCKER_IMAGE_NAME):latest .

docker-build-installation:
	docker build --target installation -t $(DOCKER_IMAGE_NAME):latest .

docker-interactive:
	docker run -it $(DOCKER_IMAGE_NAME) fish 

docker-remove:
	docker rmi -f $(DOCKER_IMAGE_NAME)

verify:
	python -c "from FLiESANN.verify import main; main()"

generate-output-dataset:
	python -c "from FLiESANN.generate_output_dataset import main; main()"

generate-static-input-dataset:
	python -c "from FLiESANN.generate_static_input_dataset import generate_static_input_dataset; generate_static_input_dataset()"

generate-FLiESANN-GEOS5FP-inputs:
	python -c "from FLiESANN.generate_FLiESANN_GEOS5FP_inputs import generate_FLiESANN_GEOS5FP_inputs; generate_FLiESANN_GEOS5FP_inputs()"

generate-input-dataset:
ifdef VARS
	python -c "from FLiESANN.generate_input_dataset import generate_input_dataset; generate_input_dataset(regenerate_variables='$(VARS)'.split())"
else
	python -c "from FLiESANN.generate_input_dataset import generate_input_dataset; generate_input_dataset()"
endif
