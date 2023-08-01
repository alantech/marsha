.PHONY: docker-build
docker-build:
	docker build -t openllamacpp .

.PHONY: docker-run
docker-run:
	docker run -p 8765:8765 --name openllamacpp openllamacpp

.PHONY: docker-stop
docker-stop:
	docker container stop openllamacpp && docker container rm openllamacpp
