# =============================================================================
#  stop_knitra.ps1  --  Stop Knitra dev server (backend + frontend)
#
#  Stops:       FastAPI backend  (port 8000)
#               NiceGUI frontend  (port 3000)
#
#  Does NOT stop:
#               PostgreSQL / Docker container  (data is preserved)
#               Ollama
#
#  Usage (from the project root):
#    .\stop_knitra.ps1
# =============================================================================

$ErrorActionPreference = "SilentlyContinue"

function Write-Step ([string]$msg) { Write-Host "  [>>] $msg" -ForegroundColor Cyan   }
function Write-Ok   ([string]$msg) { Write-Host "  [OK] $msg" -ForegroundColor Green  }
function Write-Warn ([string]$msg) { Write-Host "  [!!] $msg" -ForegroundColor Yellow }
function Write-Info ([string]$msg) { Write-Host "       $msg" -ForegroundColor DarkGray }

Write-Host ""
Write-Host "  Stopping Knitra dev server..." -ForegroundColor Magenta
Write-Host ""

# ---------------------------------------------------------------------------
# Stop every listening process on the given port
# ---------------------------------------------------------------------------
function Stop-ProcessOnPort ([int]$Port, [string]$Label) {
    Write-Step "Stopping $Label on port $Port..."

    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        Write-Warn "$Label is not listening on port $Port -- nothing to do."
        return
    }

    $targetPids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
    $killed = 0

    foreach ($pid in $targetPids) {
        # taskkill is more reliable than Stop-Process when the target was launched
        # in a different session (e.g. via Start-Process in another window).
        $result = taskkill /PID $pid /F 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Killed PID $pid."
            $killed++
        } else {
            # The process may have already exited between the port check and now.
            Write-Info "PID $pid could not be killed (may have already exited)."
        }
    }

    # Wait briefly for the OS to release the socket
    Start-Sleep -Milliseconds 700

    $remaining = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($remaining) {
        # Windows sockets sometimes linger in TIME_WAIT after a kill.
        # This clears automatically within 60 s and does not block a restart.
        Write-Warn "Port $Port still listed (likely TIME_WAIT). It clears automatically."
    } elseif ($killed -gt 0) {
        Write-Ok "$Label stopped."
    }
}

# ---------------------------------------------------------------------------
Stop-ProcessOnPort 8000 "Backend  (uvicorn)"
Stop-ProcessOnPort 3000 "Frontend (NiceGUI)"

# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
Write-Host "  Backend and frontend stopped." -ForegroundColor Green
Write-Host ""
Write-Host "  PostgreSQL is still running  (data safe)." -ForegroundColor DarkGray
Write-Host "  Ollama is still running." -ForegroundColor DarkGray
Write-Host ""
Write-Host "  To restart : .\start_knitra.ps1" -ForegroundColor DarkGray
Write-Host "  To stop DB : docker compose down" -ForegroundColor DarkGray
Write-Host "  +------------------------------------------+" -ForegroundColor DarkGray
Write-Host ""
