# dd.ps1 — nnc dev driver.
#   .\dd.ps1 run    : build Release and run nnc.exe
#   .\dd.ps1 test   : build Debug and run nnc-d.exe --test
[CmdletBinding()]
param(
	[Parameter(Position = 0)]
	[ValidateSet('run', 'test')]
	[string]$Action = 'run',

	[Parameter(ValueFromRemainingArguments = $true)]
	[string[]]$Rest
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere))
{
	throw "vswhere.exe not found at $vswhere"
}

$msbuild = & $vswhere -latest -requires Microsoft.Component.MSBuild -find 'MSBuild\**\Bin\MSBuild.exe' | Select-Object -First 1
if (-not $msbuild)
{
	throw 'MSBuild not found via vswhere.'
}

switch ($Action)
{
	'run'
	{
		& $msbuild nnc.sln /p:Configuration=Release /p:Platform=x64 /m /v:minimal /nologo
		if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
		& .\exe\nnc.exe @Rest
		exit $LASTEXITCODE
	}
	'test'
	{
		& $msbuild nnc.sln /p:Configuration=Debug /p:Platform=x64 /m /v:minimal /nologo
		if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
		& .\exe\nnc-d.exe --test @Rest
		exit $LASTEXITCODE
	}
}
