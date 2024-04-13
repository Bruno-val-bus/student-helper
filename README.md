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