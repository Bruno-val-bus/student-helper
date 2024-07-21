# Set Up
## Install requirements
- Avoid second level dependencies problems as a result from first level dependencies not pinning their second level dependencies.
- E.g. When second-level-dependency-version-ranges of one first level dependency conflict with version-ranges of another first level dependency instead of a specific version (<= instead of ==)
    Use pip-tools (https://github.com/jazzband/pip-tools) to generate requirements.in file for you (pip-tools allow you to create a file with just top-level dependencies)
    Then when pip-compile is run, pip-tools creates your requirements.txt for you with all of version compatible second-level dependencies.
- BASH COMMANDS
  - activate virtual environment `source path/to/venv/Scripts/activate`
  - install pip-tools via cmd line (running module as script) or install it via IDE
  `python -m pip install pip-tools`

  - Once installed, compile the requirements.in file to requirements.txt
  `python -m pip-compile --output-file requirements.txt requirements.in`

  - Install all packages in requirements.txt for the first installation
  `pip install -r requirements.txt`

  - or
  - Upgrade all packages in requirements.txt
  `python -m pip-sync requirements.txt`

  - Deactivate virtual environment
  `path/to/venv/Scripts/deactivate`

    
## Running Python Models with Docker Compose

This project uses Docker and Docker Compose to manage and run the Python models along with other dependent services.
Docker Compose allows you to define and run multi-container Docker applications with ease.
This will enable smoother development between different components
and also for building and deploying the entire system.

### Getting Started

To build and run the Docker containers for the models and services, follow these steps:

### Docker Compose File

In the `docker-compose.yaml` file, the services are defined as follows:

```yaml
services:
  student_helper_backend:
    #image: firstapp:latest
    build:
      context: .
    container_name: firstapp
    ports:
      - "8080:80"
    volumes:
      - ./app1_data:/app/data
    environment:
      - .env

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - /ollama:/root/.ollama
    container_name: ollama
    
```

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

docker logs <container-name>
```
You can find the container names using the docker ps command, which lists all running containers:

```sh
docker ps
```