# dd.ps1 — nnc dev driver.
#   .\dd.ps1 run              : build Release and run nnc.exe
#   .\dd.ps1 test             : build Debug and run nnc-d.exe --test
#   .\dd.ps1 download         : list the known test models
#   .\dd.ps1 download <name>  : download one (or 'all') into .\models
[CmdletBinding()]
param(
	[Parameter(Position = 0)]
	[ValidateSet('run', 'test', 'download')]
	[string]$Action = 'run',

	[Parameter(ValueFromRemainingArguments = $true)]
	[string[]]$Rest
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# Models nnc can actually load: BF16/F16/F32 straight from the mmap, or
# Q4_K/Q5_K/Q6_K decoded at load time. Q4_0/Q8_0/IQ* files will be rejected.
$Models = [ordered]@{
	'gemma-3-1b-bf16' = @{
		File = 'gemma-3-1b-it-BF16.gguf'
		Url  = 'https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-BF16.gguf'
		Size = '1.9 GB'
		Note = 'smallest end-to-end test; exercises the BF16 -> Q8_0 path'
	}
	'gemma-3-1b-q6k'  = @{
		File = 'gemma-3-1b-it-Q6_K.gguf'
		Url  = 'https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q6_K.gguf'
		Size = '1.0 GB'
		Note = 'exercises the Q6_K -> Q8_0 load-time decode'
	}
	'gemma-3-4b-q4km' = @{
		File = 'gemma-3-4b-it-Q4_K_M.gguf'
		Url  = 'https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf'
		Size = '2.5 GB'
		Note = 'exercises the Q4_K -> Q8_0 load-time decode'
	}
}

function Get-Model([string]$Key)
{
	$m = $Models[$Key]
	$dst = Join-Path 'models' $m.File
	if (Test-Path -LiteralPath $dst)
	{
		$mb = [math]::Round((Get-Item -LiteralPath $dst).Length / 1MB, 1)
		Write-Host "already present: $dst ($mb MB)"
		return
	}
	New-Item -ItemType Directory -Force -Path 'models' | Out-Null
	Write-Host "downloading $($m.File) ($($m.Size)) ..."
	$tmp = "$dst.part"
	$prev = $ProgressPreference
	$ProgressPreference = 'SilentlyContinue'
	try
	{
		Invoke-WebRequest -Uri $m.Url -OutFile $tmp
		Move-Item -LiteralPath $tmp -Destination $dst
	}
	finally
	{
		$ProgressPreference = $prev
		if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force }
	}
	$mb = [math]::Round((Get-Item -LiteralPath $dst).Length / 1MB, 1)
	Write-Host "saved $dst ($mb MB)"
}

if ($Action -eq 'download')
{
	if (-not $Rest -or $Rest.Count -eq 0)
	{
		Write-Host 'usage: .\dd.ps1 download <name|all>'
		Write-Host ''
		foreach ($k in $Models.Keys)
		{
			$m = $Models[$k]
			$have = Test-Path -LiteralPath (Join-Path 'models' $m.File)
			Write-Host ("  {0,-16} {1,-8} {2}{3}" -f $k, $m.Size, $m.Note, $(if ($have) { '  [downloaded]' } else { '' }))
		}
		exit 0
	}
	$names = if ($Rest[0] -eq 'all') { @($Models.Keys) } else { $Rest }
	foreach ($n in $names)
	{
		if (-not $Models.Contains($n))
		{
			Write-Error "unknown model '$n'. Run '.\dd.ps1 download' to list them."
		}
		Get-Model $n
	}
	exit 0
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere))
{
	throw "vswhere.exe not found at $vswhere"
}

$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $vsPath)
{
	throw 'No Visual Studio installation with the C++ toolset was found.'
}

# Ninja invokes cl.exe directly, so the MSVC x64 environment has to be in this
# process before CMake configures.
if (-not (Get-Command cl -ErrorAction SilentlyContinue))
{
	Import-Module (Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
	Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
	Set-Location -LiteralPath $PSScriptRoot
}

# Fall back to the cmake/ninja shipped with Visual Studio if they aren't on PATH.
foreach ($tool in @(@{ Name = 'cmake'; Dir = 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin' },
		@{ Name = 'ninja'; Dir = 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja' }))
{
	if (Get-Command $tool.Name -ErrorAction SilentlyContinue) { continue }
	$dir = Join-Path $vsPath $tool.Dir
	if (Test-Path -LiteralPath (Join-Path $dir "$($tool.Name).exe"))
	{
		$env:PATH = "$dir;$env:PATH"
	}
	else
	{
		throw "$($tool.Name) not found on PATH or under $vsPath."
	}
}

function Invoke-Build([string]$Preset)
{
	& cmake --preset $Preset
	if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
	& cmake --build --preset $Preset
	if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Action)
{
	'run'
	{
		Invoke-Build 'release'
		& .\exe\nnc.exe @Rest
		exit $LASTEXITCODE
	}
	'test'
	{
		Invoke-Build 'debug'
		& .\exe\nnc-d.exe --test @Rest
		exit $LASTEXITCODE
	}
}
