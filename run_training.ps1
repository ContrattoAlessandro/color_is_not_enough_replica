param([string]$Resume, [string]$OutputDirectory = 'artifacts/training')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExecutable = 'C:/Program Files/Python312/python.exe'
if ($Resume) {
    & $pythonExecutable -u -m cine train --resume $Resume --output $OutputDirectory --evaluate-after
} else {
    & $pythonExecutable -u -m cine train --output $OutputDirectory --evaluate-after
}
exit $LASTEXITCODE
