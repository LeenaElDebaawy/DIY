# =============================================================================
#  knitra.ps1  --  Knitra Development Environment Manager
#
#  Interactive menu for starting, stopping, restarting, and monitoring the
#  local Knitra development stack.
#
#  Usage:
#    .\knitra.ps1
#
#  Services managed:
#    PostgreSQL  - Docker Compose (docker-compose.yml)
#    Ollama      - checked only (start/stop handled manually)
#    Backend     - FastAPI / Uvicorn  on port 8000
#    Frontend    - NiceGUI            on port 3000
# =============================================================================

# Keep errors non-terminating by default so checks do not throw.
# Individual actions will set their own preference where needed.
$ErrorActionPreference = "SilentlyContinue"

# =============================================================================
# CONFIGURATION  --  all paths derived from the script location
# =============================================================================
$Script:Root        = $PSScriptRoot
$Script:BackendDir  = Join-Path $Script:Root "backend"
$Script:FrontendDir = Join-Path $Script:Root "frontend"
$Script:PythonExe   = Join-Path $Script:Root "backend\.venv\Scripts\python.exe"

$Script:BackendPort  = 8000
$Script:FrontendPort = 3000
$Script:OllamaUrl    = "http://localhost:11434"
$Script:FrontendUrl  = "http://localhost:3000"
$Script:BackendUrl   = "http://localhost:8000"

# =============================================================================
# LOGGING  --  consistent [TAG] output, ASCII only
# =============================================================================
function Log-Step  { param([string]$msg) Write-Host "[STEP] $msg" -ForegroundColor Cyan    }
function Log-OK    { param([string]$msg) Write-Host "[OK]   $msg" -ForegroundColor Green   }
function Log-Info  { param([string]$msg) Write-Host "[INFO] $msg" -ForegroundColor Gray    }
function Log-Warn  { param([string]$msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow  }
function Log-Error { param([string]$msg) Write-Host "[ERROR] $msg" -ForegroundColor Red    }
function Log-Blank { Write-Host "" }

# =============================================================================
# SERVICE DETECTION HELPERS
# =============================================================================

# Returns $true if a LISTEN socket exists on the given port
function Test-PortListening {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return ($null -ne $conn)
}

# Returns $true if the docker CLI is on PATH
function Test-DockerAvailable {
    return ($null -ne (Get-Command docker -ErrorAction SilentlyContinue))
}

# Returns $true if the Ollama HTTP API responds on port 11434
function Test-OllamaRunning {
    try {
        $r = Invoke-WebRequest -Uri $Script:OllamaUrl `
                               -TimeoutSec 3 `
                               -UseBasicParsing `
                               -ErrorAction Stop
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

# Returns $true if the fixly_postgres container is running.
# Falls back to a port-5433 TCP check when Docker is unavailable.
function Test-PostgresRunning {
    if (Test-DockerAvailable) {
        $status = docker inspect --format "{{.State.Running}}" fixly_postgres 2>&1
        if ($status -eq "true") { return $true }
    }
    # Fallback: check if anything is listening on 5433
    $conn = Get-NetTCPConnection -LocalPort 5433 -State Listen -ErrorAction SilentlyContinue
    return ($null -ne $conn)
}

# =============================================================================
# PROCESS MANAGEMENT HELPERS
# =============================================================================

# Launch a PowerShell command in a new detached console window.
# Uses -EncodedCommand (Base64) to avoid quoting issues with long paths.
function Start-InNewWindow {
    param([string]$Command)
    $bytes   = [System.Text.Encoding]::Unicode.GetBytes($Command)
    $encoded = [Convert]::ToBase64String($bytes)
    Start-Process powershell.exe `
        -ArgumentList "-NoExit", "-EncodedCommand", $encoded `
        -WindowStyle Normal
}

# Kill every LISTEN process on the given port.
# Uses taskkill (more reliable across sessions than Stop-Process).
function Stop-ProcessOnPort {
    param([int]$Port, [string]$Label)

    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        Log-Warn "$Label is not running on port $Port -- nothing to stop."
        return
    }

    $targetPids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
    $killed = 0

    foreach ($p in $targetPids) {
        $result = taskkill /PID $p /F 2>&1
        if ($LASTEXITCODE -eq 0) {
            Log-OK "Stopped PID $p ($Label)."
            $killed++
        } else {
            Log-Info "PID $p could not be killed (may have already exited)."
        }
    }

    Start-Sleep -Milliseconds 700

    $remaining = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($remaining) {
        Log-Warn "Port $Port still listed (TIME_WAIT is normal -- clears within 60 s)."
    } elseif ($killed -gt 0) {
        Log-OK "$Label stopped cleanly."
    }
}

# =============================================================================
# ACTION: START
# =============================================================================
function Invoke-Start {
    Log-Blank
    Write-Host "  Starting Knitra..." -ForegroundColor Magenta
    Log-Blank

    # -------------------------------------------------------------------------
    # 1. PostgreSQL
    # -------------------------------------------------------------------------
    Log-Step "PostgreSQL -- docker compose up -d db"

    if (-not (Test-DockerAvailable)) {
        Log-Warn "docker not found. Skipping PostgreSQL start."
        Log-Info "If PostgreSQL is already running on port 5433, this is fine."
    } else {
        try {
            Push-Location $Script:Root
            $output   = docker compose up -d db 2>&1
            $exitCode = $LASTEXITCODE
            Pop-Location

            if ($exitCode -eq 0) {
                Log-OK "PostgreSQL is running (fixly_postgres, port 5433)."
            } else {
                Log-Warn "docker compose exited with code $exitCode."
                Log-Info "$output"
            }
        } catch {
            Pop-Location -ErrorAction SilentlyContinue
            Log-Error "docker compose failed: $_"
        }
    }

    # -------------------------------------------------------------------------
    # 2. Ollama check
    # -------------------------------------------------------------------------
    Log-Step "Ollama -- checking $Script:OllamaUrl"

    if (Test-OllamaRunning) {
        Log-OK "Ollama is running."
    } else {
        Log-Warn "Ollama is NOT responding on port 11434."
        Log-Info "AI analysis will fail until Ollama is started."
        Log-Info "To start: open the Ollama app, or run 'ollama serve' in a terminal."
    }

    # -------------------------------------------------------------------------
    # 3. Backend (FastAPI / Uvicorn, port 8000)
    # -------------------------------------------------------------------------
    Log-Step "Backend -- port $Script:BackendPort"

    if (Test-PortListening $Script:BackendPort) {
        Log-OK "Port $Script:BackendPort already in use -- backend is running. Skipping."
    } elseif (-not (Test-Path $Script:PythonExe)) {
        Log-Error "Python venv not found: $Script:PythonExe"
        Log-Info "Run: cd backend; python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    } else {
        $BackendCmd = @"
`$Host.UI.RawUI.WindowTitle = 'Knitra Backend (port 8000)'
Write-Host ' Knitra Backend -- http://localhost:8000 ' -ForegroundColor White -BackgroundColor DarkBlue
Write-Host ''
Set-Location '$Script:BackendDir'
& '$Script:PythonExe' -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"@
        Start-InNewWindow $BackendCmd
        Log-OK "Backend window launched (uvicorn, port 8000, --reload on)."
        Log-Info "Waiting 3 s for backend to bind..."
        Start-Sleep -Seconds 3
    }

    # -------------------------------------------------------------------------
    # 4. Frontend (NiceGUI, port 3000)
    # -------------------------------------------------------------------------
    Log-Step "Frontend -- port $Script:FrontendPort"

    if (Test-PortListening $Script:FrontendPort) {
        Log-OK "Port $Script:FrontendPort already in use -- frontend is running. Skipping."
    } elseif (-not (Test-Path $Script:PythonExe)) {
        Log-Error "Python venv not found: $Script:PythonExe"
    } else {
        # frontend/ has no separate venv -- it reuses backend/.venv
        $FrontendCmd = @"
`$Host.UI.RawUI.WindowTitle = 'Knitra Frontend (port 3000)'
Write-Host ' Knitra Frontend -- http://localhost:3000 ' -ForegroundColor White -BackgroundColor DarkMagenta
Write-Host ''
Set-Location '$Script:FrontendDir'
& '$Script:PythonExe' main.py
"@
        Start-InNewWindow $FrontendCmd
        Log-OK "Frontend window launched (NiceGUI, port 3000)."
    }

    # -------------------------------------------------------------------------
    # 5. Wait for frontend, then open browser
    # -------------------------------------------------------------------------
    Log-Step "Waiting for frontend to respond..."

    $frontendReady = $false
    $maxWait       = 60
    $poll          = 2

    for ($elapsed = 0; $elapsed -lt $maxWait; $elapsed += $poll) {
        Start-Sleep -Seconds $poll
        try {
            $r = Invoke-WebRequest -Uri $Script:FrontendUrl `
                                   -TimeoutSec 2 `
                                   -UseBasicParsing `
                                   -ErrorAction Stop
            if ($r.StatusCode -eq 200) {
                $frontendReady = $true
                break
            }
        } catch { }
        Log-Info "... ${elapsed}s elapsed"
    }

    if ($frontendReady) {
        Log-OK "Frontend ready. Opening browser..."
        Start-Process $Script:FrontendUrl
    } else {
        Log-Warn "Frontend did not respond within ${maxWait}s."
        Log-Info "Open manually: $Script:FrontendUrl"
        Log-Info "Check the Frontend window for Python errors."
    }

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    Log-Blank
    Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
    Write-Host "  Knitra is running." -ForegroundColor Green
    Log-Blank
    Write-Host "  App      : $Script:FrontendUrl" -ForegroundColor White
    Write-Host "  API      : $Script:BackendUrl"  -ForegroundColor White
    Write-Host "  Docs     : $Script:BackendUrl/docs" -ForegroundColor White
    Write-Host "  Database : localhost:5433  (fixly_postgres)" -ForegroundColor White
    Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
    Log-Blank
}

# =============================================================================
# ACTION: STOP
# =============================================================================
function Invoke-Stop {
    Log-Blank
    Write-Host "  Stopping Knitra..." -ForegroundColor Magenta
    Log-Blank

    Log-Step "Stopping Frontend (NiceGUI, port $Script:FrontendPort)..."
    Stop-ProcessOnPort $Script:FrontendPort "Frontend (NiceGUI)"

    Log-Step "Stopping Backend (uvicorn, port $Script:BackendPort)..."
    Stop-ProcessOnPort $Script:BackendPort  "Backend  (uvicorn)"

    Log-Blank
    Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
    Write-Host "  Backend and frontend stopped." -ForegroundColor Green
    Log-Blank
    Write-Host "  PostgreSQL is still running  (data safe)." -ForegroundColor DarkGray
    Write-Host "  Ollama is still running." -ForegroundColor DarkGray
    Log-Blank
    Write-Host "  To stop the database: docker compose down" -ForegroundColor DarkGray
    Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
    Log-Blank
}

# =============================================================================
# ACTION: RESTART
# =============================================================================
function Invoke-Restart {
    Log-Blank
    Write-Host "  Restarting Knitra..." -ForegroundColor Magenta
    Log-Blank

    Invoke-Stop

    Log-Info "Waiting 3 s before restart..."
    Start-Sleep -Seconds 3

    Invoke-Start
}

# =============================================================================
# ACTION: STATUS CHECK
# =============================================================================
function Invoke-Status {
    Log-Blank
    Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
    Write-Host "  Knitra Service Status" -ForegroundColor Cyan
    Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
    Log-Blank

    # PostgreSQL
    if (Test-PostgresRunning) {
        Write-Host "  PostgreSQL (5433)  : " -NoNewline
        Write-Host "Running" -ForegroundColor Green
    } else {
        Write-Host "  PostgreSQL (5433)  : " -NoNewline
        Write-Host "Stopped" -ForegroundColor Red
    }

    # Ollama
    if (Test-OllamaRunning) {
        Write-Host "  Ollama (11434)     : " -NoNewline
        Write-Host "Running" -ForegroundColor Green
    } else {
        Write-Host "  Ollama (11434)     : " -NoNewline
        Write-Host "Stopped" -ForegroundColor Red
        Write-Host "                       (AI analysis will not work)" -ForegroundColor Yellow
    }

    # Backend
    if (Test-PortListening $Script:BackendPort) {
        Write-Host "  Backend  ($Script:BackendPort)  : " -NoNewline
        Write-Host "Running  --> $Script:BackendUrl" -ForegroundColor Green
    } else {
        Write-Host "  Backend  ($Script:BackendPort)  : " -NoNewline
        Write-Host "Stopped" -ForegroundColor Red
    }

    # Frontend
    if (Test-PortListening $Script:FrontendPort) {
        Write-Host "  Frontend ($Script:FrontendPort)  : " -NoNewline
        Write-Host "Running  --> $Script:FrontendUrl" -ForegroundColor Green
    } else {
        Write-Host "  Frontend ($Script:FrontendPort)  : " -NoNewline
        Write-Host "Stopped" -ForegroundColor Red
    }

    Log-Blank

    # Quick reachability check for running services
    if (Test-PortListening $Script:BackendPort) {
        try {
            $rb = Invoke-WebRequest -Uri "$Script:BackendUrl/health" `
                                    -TimeoutSec 3 `
                                    -UseBasicParsing `
                                    -ErrorAction Stop
            Log-Info "Backend health check : HTTP $($rb.StatusCode) OK"
        } catch {
            Log-Warn "Backend is listening but health endpoint did not respond."
        }
    }

    if (Test-PortListening $Script:FrontendPort) {
        try {
            $rf = Invoke-WebRequest -Uri $Script:FrontendUrl `
                                    -TimeoutSec 3 `
                                    -UseBasicParsing `
                                    -ErrorAction Stop
            Log-Info "Frontend reachability: HTTP $($rf.StatusCode) OK"
        } catch {
            Log-Warn "Frontend is listening but did not return HTTP 200."
        }
    }

    Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
    Log-Blank
}

# =============================================================================
# ACTION: OPEN BROWSER
# =============================================================================
function Invoke-OpenBrowser {
    Log-Blank
    if (Test-PortListening $Script:FrontendPort) {
        Log-OK "Opening browser at $Script:FrontendUrl"
        Start-Process $Script:FrontendUrl
    } else {
        Log-Warn "Frontend is NOT running on port $Script:FrontendPort."
        Log-Info "Start the stack first (menu option 1), then try again."
    }
    Log-Blank
}

# =============================================================================
# MENU  --  interactive loop, returns only when the user chooses Exit
# =============================================================================
function Show-Menu {
    while ($true) {
        Clear-Host
        Write-Host ""
        Write-Host "  ========================================" -ForegroundColor Cyan
        Write-Host "  Knitra Development Environment Manager" -ForegroundColor Cyan
        Write-Host "  ========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  [1] Start Knitra"   -ForegroundColor White
        Write-Host "  [2] Stop Knitra"    -ForegroundColor White
        Write-Host "  [3] Restart Knitra" -ForegroundColor White
        Write-Host "  [4] Status Check"   -ForegroundColor White
        Write-Host "  [5] Open Browser"   -ForegroundColor White
        Write-Host "  [6] Exit"           -ForegroundColor DarkGray
        Write-Host ""

        # Show a one-line live status hint so the user knows the current state
        # without running a full status check
        $bRunning = Test-PortListening $Script:BackendPort
        $fRunning = Test-PortListening $Script:FrontendPort
        $bLabel   = if ($bRunning) { "UP" } else { "DOWN" }
        $fLabel   = if ($fRunning) { "UP" } else { "DOWN" }
        $bColour  = if ($bRunning) { "Green" } else { "Red" }
        $fColour  = if ($fRunning) { "Green" } else { "Red" }

        Write-Host "  Current : Backend " -NoNewline -ForegroundColor DarkGray
        Write-Host $bLabel -NoNewline -ForegroundColor $bColour
        Write-Host " | Frontend " -NoNewline -ForegroundColor DarkGray
        Write-Host $fLabel -ForegroundColor $fColour
        Write-Host ""

        $choice = Read-Host "  Enter choice (1-6)"

        switch ($choice.Trim()) {
            "1" { Invoke-Start;       Read-Host "  [Press Enter to return to menu]" }
            "2" { Invoke-Stop;        Read-Host "  [Press Enter to return to menu]" }
            "3" { Invoke-Restart;     Read-Host "  [Press Enter to return to menu]" }
            "4" { Invoke-Status;      Read-Host "  [Press Enter to return to menu]" }
            "5" { Invoke-OpenBrowser; Read-Host "  [Press Enter to return to menu]" }
            "6" {
                Log-Blank
                Write-Host "  Bye!" -ForegroundColor DarkGray
                Log-Blank
                return
            }
            default {
                Write-Host "  Invalid choice. Enter a number from 1 to 6." -ForegroundColor Yellow
                Start-Sleep -Seconds 1
            }
        }
    }
}

# =============================================================================
# ENTRY POINT
# Only call Show-Menu when the script is executed directly.
# When dot-sourced (. .\knitra.ps1) the functions are loaded without launching
# the interactive menu -- useful for testing individual functions in a shell.
# =============================================================================
if ($MyInvocation.InvocationName -ne '.') {
    Show-Menu
}
