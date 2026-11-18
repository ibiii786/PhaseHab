# Setup (do once)

Open PowerShell in `D:\Projects\FYP\code` and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/AIM-Harvard/pyradiomics
python -c "import radiomics, SimpleITK, nibabel, xgboost, shap; print('all good')"
```

If `Activate.ps1` is blocked: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

If the PyRadiomics line fails with a compiler error: Visual Studio 2022 Community with MSVC 14.44 is already installed on this PC, so open "x64 Native Tools Command Prompt for VS 2022", activate the venv there, and run the line again.

Or just open Claude Code in this folder and ask it to "do the setup in SETUP.md".
