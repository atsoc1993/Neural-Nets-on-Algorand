# Setup

## 1. Python
- Download [Python](https://www.python.org/downloads/) if not already available
- Create a virtual environment
`python -m venv venv`
- If `(venv)` not showing at left-hand side of terminal after virtual environment creation, use: 
-- Windows Powershell: `./venv/scripts/activate.ps1`
-- Mac/Linux: `source .venv/bin/activate`

## 2. Libraries

### Fast
- Use `pip install -r requirements.txt`

### Manual
- Install Algokit: `pip install algokit`
- Install Algokit Client Generator: `pip install algokit-client-generator`

## 3. Environment Setup
- Install and **Start** [Docker Desktop](https://docs.docker.com/engine/install/)
- Start Localnet Instance: `algokit localnet start`
- Show Localnet Accounts: `algokit goal account list`
- Export any of the displayed accounts with large algo balance:

`algokit goal account export -a <some address from previous list>`

- Create a `.env` file, set secrets to `ADDRESS` and `MNEMONIC` in this file to the address and mnemonic provided from previous command

