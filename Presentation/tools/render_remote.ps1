# Render scenes on the Windows render machine, all in parallel. Called by ../render.sh over ssh.
#   render_remote.ps1 -Scenes A,B,C [-Quality l|m|h|p|k]
# Each scene gets its own media folder (media_scenes\<scene>): parallel processes sharing one
# media\texts collide on Windows file locks. The LaTeX cache (media\Tex) comes from the Linux
# machine, since this one has no LaTeX.
param([Parameter(Mandatory = $true)][string[]]$Scenes, [string]$Quality = "")
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force logs | Out-Null
$t0 = Get-Date
$procs = foreach ($s in $Scenes) {
  $md = "media_scenes\$s"
  New-Item -ItemType Directory -Force $md | Out-Null
  robocopy media\Tex "$md\Tex" /E /NFL /NDL /NJH /NJS /NC /NS /NP | Out-Null
  $a = @("render")
  if ($Quality) { $a += "-q$Quality" }
  $a += @("--media_dir", $md, "slides.py", $s)
  Start-Process -FilePath venv\Scripts\manim-slides.exe -ArgumentList $a `
    -RedirectStandardOutput "logs\$s.out" -RedirectStandardError "logs\$s.err" -NoNewWindow -PassThru
}
$procs | Wait-Process
$bad = @()
for ($i = 0; $i -lt $Scenes.Count; $i++) { if ($procs[$i].ExitCode -ne 0) { $bad += $Scenes[$i] } }
"rendered $($Scenes.Count) scene(s) in $([int]((Get-Date) - $t0).TotalSeconds) s"
if ($bad.Count) {
  foreach ($s in $bad) { "---- $s (logs\$s.err) ----"; Get-Content "logs\$s.err" -Tail 15 }
  "FAILED: $($bad -join ' ')"
  exit 1
}
exit 0
