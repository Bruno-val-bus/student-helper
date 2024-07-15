# Install requirements
## Philosophy
- Avoid second level dependencies problems as a result from first level dependencies not pinning their second level dependencies.
- E.g. When second-level-dependency-version-ranges of one first level dependency conflict with version-ranges of another first level dependency instead of a specific version (<= instead of ==)
    Use pip-tools (https://github.com/jazzband/pip-tools) to generate requirements.in file for you (pip-tools allow you to create a file with just top-level dependencies)
    Then when pip-compile is run, pip-tools creates your requirements.txt for you with all of version compatible second-level dependencies.
## Install
- Project requires python version 3.11
- Create virtual environment `python -m venv .venv`
- Activate virtual environment `.venv/Scripts/activate`
- Install pip-tools `python -m pip install pip-tools`
- Compile requirements.in file to requirements.txt. This will create a new requirements.txt if it does not exist or update an existing one. In the latter case (e.g. when adding new libs), existing dependencies will remain in their pinned version, unless there are conflicts with the new libs. If so, it will try to resolve the conflicts. 
  `python -m pip-compile --output-file requirements.txt requirements.in`
- Install all packages in requirements.txt for the first installation
  `pip install -r requirements.txt`
- Upgrade all packages in requirements.txt for subsequent installations (e.g. when adding new libs)
  `python -m pip-sync requirements.txt`
- Deactivate virtual environment
  `.venv/Scripts/deactivate`
  
# Set OpenAi Key
- Create .env file in root
- Write `OPENAI_API_KEY = "my-key"`