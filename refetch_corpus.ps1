# ==============================================================
# Re-fetch the PMC corpus after the <xref> tail-truncation fix
# ==============================================================
#
#   .\refetch_corpus.ps1              # dry run - shows what would change
#   .\refetch_corpus.ps1 -Confirm     # actually deletes and re-fetches
#
# WHY THIS IS NEEDED:
# _strip_noise() removed <xref> citation markers with parent.remove(child).
# In ElementTree the text FOLLOWING an element lives on that element as
# `.tail`, so removing it also deleted the rest of the paragraph. Every PMC
# article was truncated at its first reference marker.
#
# Re-embedding cannot recover this - the text was never stored. It has to
# come back from PMC, which means deleting the truncated copies first
# (ingestion is idempotent on title, so it would otherwise skip them).
#
# ASCII ONLY, ON PURPOSE.
# Windows PowerShell 5.1 reads .ps1 files as ANSI unless they carry a UTF-8
# BOM. A UTF-8 em dash then decodes to three cp1252 characters, one of which
# is U+201D - a smart quote that PowerShell accepts as a string delimiter.
# The result is "Unexpected token '}'" pointing at a line that is perfectly
# valid. Keeping this file to ASCII removes the dependency entirely.
#
# NOTE: uses Invoke-RestMethod, not curl. In PowerShell `curl` is an alias
# for Invoke-WebRequest, which does not accept -H or -d.
# ==============================================================

param(
    [switch]$Confirm,
    # Override the safety check that refuses to delete an already-repaired
    # corpus. You should almost never need this.
    [switch]$Force,
    [string]$Api = "http://localhost:8000",
    [int]$MaxPerTopic = 10
)

$ErrorActionPreference = "Stop"

function Say($msg)  { Write-Host "`n$msg" -ForegroundColor White }
function Ok($msg)   { Write-Host "  [ok]   $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  [warn] $msg" -ForegroundColor Yellow }
function Bad($msg)  { Write-Host "  [FAIL] $msg" -ForegroundColor Red }
function Dim($msg)  { Write-Host "         $msg" -ForegroundColor DarkGray }

# -- Reachable? ------------------------------------------------
Say "Checking the backend"
try {
    $health = Invoke-RestMethod "$Api/api/v1/health" -TimeoutSec 15
    Ok "backend healthy (database: $($health.components.database))"
} catch {
    Bad "Backend unreachable at $Api"
    Dim "Start it:  docker-compose up -d"
    exit 1
}

# -- What is there now? ----------------------------------------
Say "Inspecting the current PMC corpus"

$docs = @()
$page = 1
do {
    $url = "$Api/api/v1/knowledge/documents?source_type=pmc_open_access&page=$page&page_size=100"
    $batch = Invoke-RestMethod $url -TimeoutSec 30
    if ($batch.documents) { $docs += $batch.documents }
    $page = $page + 1
} while ($batch.documents.Count -eq 100)

$mean = 0
if ($docs.Count -eq 0) {
    Warn "No PMC articles found - nothing to delete. Will just fetch."
} else {
    $chunks = ($docs | Measure-Object -Property chunk_count -Sum).Sum
    if (-not $chunks) { $chunks = 0 }
    $mean = [math]::Round($chunks / $docs.Count, 1)

    Ok "$($docs.Count) articles, $chunks chunks"
    Write-Host ""
    Write-Host "         MEAN CHUNKS PER ARTICLE: $mean" -ForegroundColor Cyan
    Write-Host ""

    # A full-text radiology article runs to many thousands of characters. At
    # CHUNK_SIZE=512 that is tens of chunks. Single digits means the text was
    # cut off at the first citation marker in each paragraph.
    if ($mean -lt 8) {
        Warn "Low - these look truncated, which is exactly what this fixes."
        Dim "Write this number down; it is your before/after evidence."
    } else {
        Warn "Looks healthy - these may already have been re-fetched."
    }
}

# -- Dry run stops here ----------------------------------------
if (-not $Confirm) {
    Say "DRY RUN - nothing was changed."
    Dim "Would delete $($docs.Count) PMC articles, then re-fetch from NCBI."
    Write-Host "         Re-run with -Confirm to proceed:" -ForegroundColor DarkGray
    Write-Host "             .\refetch_corpus.ps1 -Confirm" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}

# -- Refuse to delete a corpus that is already healthy ---------
# This script exists to repair truncated articles ONCE. Running it a second
# time after a successful repair would delete good articles and re-download
# them over a flaky connection, for no benefit and considerable risk.
#
# The mean is the discriminator: truncated articles produce single-digit
# chunk counts, repaired ones tens. If the corpus already looks healthy,
# stop and point at the endpoint that ADDS without deleting.
if ($docs.Count -gt 0 -and $mean -ge 8) {
    Say "REFUSING TO DELETE - the corpus already looks repaired."
    Dim "Mean is $mean chunks/article; truncated articles score under 8."
    Write-Host ""
    Dim "To ADD more articles without touching what you have, ingestion is"
    Dim "idempotent on title, so just call fetch-pmc directly:"
    Write-Host ""
    Write-Host "    Invoke-RestMethod $Api/api/v1/knowledge/fetch-pmc ``" -ForegroundColor Cyan
    Write-Host "      -Method Post -ContentType application/json ``" -ForegroundColor Cyan
    Write-Host "      -Body '{\"max_per_topic\": 25}'" -ForegroundColor Cyan
    Write-Host ""
    Dim "If you really do want to wipe and re-download, add -Force."
    Write-Host ""
    if (-not $Force) { exit 0 }
    Warn "-Force given; deleting anyway."
}

# -- Delete ----------------------------------------------------
if ($docs.Count -gt 0) {
    Say "Deleting truncated articles"
    $deleted = 0
    $failed  = 0
    foreach ($doc in $docs) {
        try {
            Invoke-RestMethod "$Api/api/v1/knowledge/documents/$($doc.id)" `
                -Method Delete -TimeoutSec 60 | Out-Null
            $deleted = $deleted + 1
            if ($deleted % 25 -eq 0) { Dim "$deleted deleted..." }
        } catch {
            $failed = $failed + 1
        }
    }
    Ok "$deleted deleted"
    if ($failed -gt 0) { Warn "$failed could not be deleted - re-run to retry" }
}

# -- Re-fetch --------------------------------------------------
Say "Starting the re-fetch"
$body = @{ max_per_topic = $MaxPerTopic } | ConvertTo-Json -Compress

try {
    $resp = Invoke-RestMethod "$Api/api/v1/knowledge/fetch-pmc" `
        -Method Post -ContentType "application/json" -Body $body -TimeoutSec 60
    Ok $resp.message
} catch {
    Bad "fetch-pmc failed: $_"
    exit 1
}

Say "Running in the background. Watch it:"
Write-Host "             docker-compose logs -f backend" -ForegroundColor Cyan
Write-Host ""
Dim "When it settles, run this script again (no -Confirm) to see the"
Dim "new mean. It should be much higher than $mean."
Write-Host ""
Dim "On a flaky connection expect partial failures. Re-running tops up"
Dim "rather than duplicating, so run it again if the count looks short."
Write-Host ""
