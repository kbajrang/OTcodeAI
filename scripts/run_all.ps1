$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")

$venvPython = Join-Path $projectRoot "venv\\Scripts\\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
}

$defaultConnectTimeout = 10
$defaultReadTimeout = 0  # 0 => no read timeout

if (-not $env:OLLAMA_CONNECT_TIMEOUT_S) {
    $env:OLLAMA_CONNECT_TIMEOUT_S = "$defaultConnectTimeout"
} elseif ([int]$env:OLLAMA_CONNECT_TIMEOUT_S -lt 1) {
    $env:OLLAMA_CONNECT_TIMEOUT_S = "$defaultConnectTimeout"
}

if (-not $env:OLLAMA_READ_TIMEOUT_S) {
    $env:OLLAMA_READ_TIMEOUT_S = "$defaultReadTimeout"
} else {
    try {
        $readTimeout = [int]$env:OLLAMA_READ_TIMEOUT_S
        if ($readTimeout -le 120) {
            $env:OLLAMA_READ_TIMEOUT_S = "$defaultReadTimeout"
        }
    } catch {
        $env:OLLAMA_READ_TIMEOUT_S = "$defaultReadTimeout"
    }
}

if (-not $env:OLLAMA_STREAM) {
    $env:OLLAMA_STREAM = "true"
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
