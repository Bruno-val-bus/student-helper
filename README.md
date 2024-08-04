# Set Up
## Install requirements
- Avoid second level dependencies problems as a result from first level dependencies not pinning their second level dependencies.
- E.g. When second-level-dependency-version-ranges of one first level dependency conflict with version-ranges of another first level dependency instead of a specific version (<= instead of ==)
    Use pip-tools (https://github.com/jazzband/pip-tools) to generate requirements.in file for you (pip-tools allow you to create a file with just top-level dependencies)
    Then when pip-compile is run, pip-tools creates your requirements.txt for you with all of version compatible second-level dependencies.
- BASH COMMANDS
  - Activate virtual environment for bash or Windows shell
  ```shell
  source ./.venv/Scripts/activate
  - ```
  ```shell
  ./.venv/Scripts/activate
  - ```
  - Install pip-tools via cmd line (running module as script) or install it via IDE
  ```shell
  pip install pip-tools
  - ```

  - Once installed, compile `requirements.in` file to `requirements.txt`
  ```shell
  pip-compile --output-file requirements.txt requirements.in
  - ```
  - For development, compile also `requirements-dev.in` file to `requirements-dev.txt`
  ```shell
  pip-compile --output-file requirements-dev.txt requirements-dev.in requirements.in
  - ```

  - Install all packages in `requirements.txt` or `requirements-dev.txt` (for development) for the first installation
  ```shell
  pip install -r requirements.txt
  - ```
    ```shell
  pip install -r requirements-dev.txt
  - ```

  - Upgrade all packages in `requirements.txt` or `requirements-dev.txt` (for development) after first installation
  ```shell
  pip-sync requirements.txt
  - ```
    ```shell
  pip-sync requirements-dev.txt
  - ```
  
  - Deactivate virtual environment for bash or Windows shell
  ```shell
  source ./.venv/Scripts/deactivate
  - ```
  ```shell
  ./.venv/Scripts/deactivate
  - ```

    
## Running Python Models with Docker Compose

This project uses Docker and Docker Compose to manage and run the Python models along with other dependent services.
Docker Compose allows you to define and run multi-container Docker applications with ease.
This will enable smoother development between different components
and also for building and deploying the entire system.

### Getting Started

To build and run the Docker containers for the models and services,
you will need [docker](https://docs.docker.com/engine/install/)

### Using LOCAL_OLLAMA_LLAMA3 model

- To only run Ollama inside a container, run the `ollama` container from the dedicated `docker-compose.yaml` file:

```shell
docker compose -f docker-compose.yaml up ollama 
```

- In order to run specific models, these first have to be downloaded. As this setup uses the `LLAMA3:8b` model,
the following model needs to be downloaded as follows, before scripts can be run over it:

```shell
docker exec -it ollama ollama run llama3:8b
```
- The aforementioned docker cmd executes ``ollama run llama3:8b`` inside the running container `ollama`

After it is downloaded, you can now interact with ollama directly out of your python script. Useful for development.



### Using an entirely dockerized environment

In the `docker-compose.yaml` file, the services are defined as follows:

```yaml
services:
  student_helper_backend:
    build:
      context: .
    container_name: student_helper_server
    networks:
      - student-helper-net
    volumes:
      - ./app1_data:/app/data
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    networks:
      - student-helper-net
    volumes:
      - /ollama:/root/.ollama

networks:
  student-helper-net:
    driver: bridge
    
```
where the student_helper_server is the build docker container from our setup in ``.\Dockerfile``.


### Build the Docker Images:

First, navigate to the project directory.

Use the following command in the CLI to start all services in detached mode.
This runs the rebuilds and runs containers locally in the background, allowing you to continue using your terminal:

```sh
docker-compose up --build -d
```
Running the `--build` flag rebuilds your local python image, so that the changes you made are considered.
No need to run the flag if you have not changed anything during development (i.e. you want to test the integration into
another service).
Running in detached mode prevents the logs of all containers from flooding your terminal. The -d flag stands for "detached mode."

### View Logs for a Specific Container:

To view the logs for a specific container, use the docker logs command followed by the container name:

```sh
docker logs -f <container-name>
```
You can find the container names using the docker ps command, which lists all running containers:

```sh
docker ps
```

## Stopping docker containers 
- To stop all containers, run

```shell
docker compose down
```
- To stop and then remove certain containers, do the following:

```shell
docker container stop <container name>
docker container rm <container name>
```

