# Environment Snapshots

This folder records the `hydra_env` package state used for the measurement and analysis scripts.

- `hydra_env.environment.yml` is the readable Conda export. Recreate with:

  ```powershell
  C:\ProgramData\miniforge3\Scripts\conda.exe env create -f environments\hydra_env.environment.yml
  ```

- `hydra_env.conda-explicit-win-64.txt` is the exact Windows Conda package spec. Recreate most exactly on Windows with:

  ```powershell
  C:\ProgramData\miniforge3\Scripts\conda.exe create -n hydra_env_rebuild --file environments\hydra_env.conda-explicit-win-64.txt
  ```

- `hydra_env.requirements-pip.txt` is a separate `pip freeze --all` snapshot for quick inspection of pip packages.

Refresh all three files with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\export-hydra-env.ps1
```
