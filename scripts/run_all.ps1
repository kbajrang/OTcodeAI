$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")

$venvPython = Join-Path $projectRoot "venv\\Scripts\\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
}

$uvicornCmd = "& `"$venvPython`" -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000"
$streamlitCmd = "& `"$venvPython`" -m streamlit run src/ui/streamlit_app.py"

Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location `"$projectRoot`"; $uvicornCmd"
) -WorkingDirectory $projectRoot

Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location `"$projectRoot`"; $streamlitCmd"
) -WorkingDirectory $projectRoot
