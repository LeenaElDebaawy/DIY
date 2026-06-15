# =============================================================================
#  start_knitra.ps1  --  Knitra local development stack launcher
#
#  What this script does:
#    1. Starts PostgreSQL via Docker Compose  (if not already running)
#    2. Checks that Ollama is reachable       (warns if not; does not exit)
#    3. Starts the FastAPI backend            (port 8000, new window)
#    4. Starts the NiceGUI frontend           (port 3000, new window)
#    5. Opens the browser when the frontend responds
#
#  Usage (from the project root):
#    .\start_knitra.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

# Resolve paths relative to this script's location (project root)
$Root        = $PSScriptRoot
$BackendDir  = Join-Path $Root "backend"
$FrontendDir = Join-Path $Root "frontend"
$PythonExe   = Join-Path $Root "backend\.venv\Scripts\python.exe"

# ---------------------------------------------------------------------------
# Helper functions (ASCII only -- Unicode chars break PS 5.1 in some locales)
# ---------------------------------------------------------------------------
function Write-Step ([string]$msg) { Write-Host "  [>>] $msg" -ForegroundColor Cyan    }
function Write-Ok   ([string]$msg) { Write-Host "  [OK] $msg" -ForegroundColor Green   }
function Write-Warn ([string]$msg) { Write-Host "  [!!] $msg" -ForegroundColor Yellow  }
function Write-Fail ([string]$msg) { Write-Host "  [XX] $msg" -ForegroundColor Red     }
function Write-Info ([string]$msg) { Write-Host "       $msg" -ForegroundColor DarkGray }

function Test-PortListening ([int]$Port) {
    $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return ($null -ne $c)
}

# Launch a detached PowerShell window.
# Using -EncodedCommand avoids quoting problems with long paths.
function Start-InNewWindow ([string]$Command) {
    $bytes   = [System.Text.Encoding]::Unicode.GetBytes($Command)
    $encoded = [Convert]::ToBase64String($bytes)
    Start-Process powershell.exe `
        -ArgumentList "-NoExit", "-EncodedCommand", $encoded `
        -WindowStyle Normal
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "  +------------------------------------+" -ForegroundColor Magenta
Write-Host "  |  Knitra -- Dev Stack Launcher      |" -ForegroundColor Magenta
Write-Host "  +------------------------------------+" -ForegroundColor Magenta
Write-Host ""

# ===========================================================================
# STEP 1 -- PostgreSQL via Docker Compose
# ===========================================================================
Write-Step "PostgreSQL -- docker compose up -d db"

$dockerAvailable = $null -ne (Get-Command docker -ErrorAction SilentlyContinue)
if (-not $dockerAvailable) {
    Write-Warn "docker not found -- skipping PostgreSQL start."
    Write-Info "If PostgreSQL is already running on port 5433, analysis will still work."
} else {
    try {
        Push-Location $Root
        $output = docker compose up -d db 2>&1
        $exitCode = $LASTEXITCODE
        Pop-Location
        if ($exitCode -eq 0) {
            Write-Ok "PostgreSQL container is running (fixly_postgres on port 5433)."
        } else {
            Write-Warn "docker compose returned exit code $exitCode."
            Write-Info "$output"
        }
    } catch {
        Pop-Location -ErrorAction SilentlyContinue
        Write-Warn "docker compose failed: $_"
    }
}

# ===========================================================================
# STEP 2 -- Ollama check
# ===========================================================================
Write-Step "Ollama -- checking http://localhost:11434"
$ollamaOk = $false
try {
    $r = Invoke-WebRequest -Uri "http://localhost:11434" `
                           -TimeoutSec 3 `
                           -UseBasicParsing `
                           -ErrorAction Stop
    $ollamaOk = $true
    Write-Ok "Ollama is running."
} catch {
    Write-Warn "Ollama is NOT responding on port 11434."
    Write-Info "AI analysis will fail until Ollama is started."
    Write-Info "Start it: open the Ollama desktop app, or run 'ollama serve'."
}

# ===========================================================================
# STEP 3 -- FastAPI backend (port 8000)
# ===========================================================================
Write-Step "Backend -- port 8000"

if (Test-PortListening 8000) {
    Write-Ok "Port 8000 is already in use -- backend appears to be running. Skipping."
} else {
    if (-not (Test-Path $PythonExe)) {
        Write-Fail "Python venv not found: $PythonExe"
        Write-Info "Set up the venv: cd backend; python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    } else {
        # Build the command string for the new window.
        # Single-quote paths inside the here-string so PowerShell does not
        # expand them a second time when it decodes the Base64 command.
        $BackendCmd = @"
`$Host.UI.RawUI.WindowTitle = 'Knitra Backend (8000)'
Write-Host ' Knitra Backend -- http://localhost:8000 ' -ForegroundColor White -BackgroundColor DarkBlue
Write-Host ''
Set-Location '$BackendDir'
& '$PythonExe' -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"@
        Start-InNewWindow $BackendCmd
        Write-Ok "Backend window launched (uvicorn, port 8000, --reload on)."

        # Give uvicorn a moment to bind before the frontend starts making API calls
        Write-Info "Waiting 3 s for backend to bind..."
        Start-Sleep -Seconds 3
    }
}

# ===========================================================================
# STEP 4 -- NiceGUI frontend (port 3000)
# ===========================================================================
Write-Step "Frontend -- port 3000"

if (Test-PortListening 3000) {
    Write-Ok "Port 3000 is already in use -- frontend appears to be running. Skipping."
} else {
    # frontend/ has no separate venv -- it reuses backend/.venv
    $FrontendCmd = @"
`$Host.UI.RawUI.WindowTitle = 'Knitra Frontend (3000)'
Write-Host ' Knitra Frontend -- http://localhost:3000 ' -ForegroundColor White -BackgroundColor DarkMagenta
Write-Host ''
Set-Location '$FrontendDir'
& '$PythonExe' main.py
"@
    Start-InNewWindow $FrontendCmd
    Write-Ok "Frontend window launched (NiceGUI, port 3000)."
}

# ===========================================================================
# STEP 5 -- Wait for frontend, then open browser
# ===========================================================================
Write-Step "Waiting for frontend to respond..."

$frontendReady = $false
$maxWait       = 60   # seconds
$poll          = 2    # seconds between checks

for ($elapsed = 0; $elapsed -lt $maxWait; $elapsed += $poll) {
    Start-Sleep -Seconds $poll
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:3000" `
                               -TimeoutSec 2 `
                               -UseBasicParsing `
                               -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            $frontendReady = $true
            break
        }
    } catch {
        # Not ready yet -- keep waiting
    }
    Write-Info "... ${elapsed}s elapsed"
}

if ($frontendReady) {
    Write-Ok "Frontend ready. Opening browser at http://localhost:3000"
    Start-Process "http://localhost:3000"
} else {
    Write-Warn "Frontend did not respond within ${maxWait}s."
    Write-Info "Open it manually: http://localhost:3000"
    Write-Info "Check the Frontend window for Python errors."
}

# ===========================================================================
# Summary
# ===========================================================================
Write-Host ""
Write-Host "  +-----------------------------------------+" -ForegroundColor DarkGray
Write-Host "  Knitra dev stack is running" -ForegroundColor Green
Write-Host ""
Write-Host "  App      : http://localhost:3000" -ForegroundColor White
Write-Host "  API      : http://localhost:8000" -ForegroundColor White
Write-Host "  API docs : http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Database : localhost:5433  (fixly_postgres)" -ForegroundColor White
if ($ollamaOk) {
    Write-Host "  Ollama   : http://localhost:11434" -ForegroundColor White
} else {
    Write-Host "  Ollama   : NOT RUNNING  -- start this for AI analysis" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "  To stop: .\stop_knitra.ps1" -ForegroundColor DarkGray
Write-Host "  +-----------------------------------------+" -ForegroundColor DarkGray
Write-Host ""
