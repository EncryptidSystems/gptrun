@echo off
setlocal EnableExtensions
rem dawnrt.exe: the resident Dawn device ModelRun drives over stdin/stdout.
rem Uses the same Dawn snapshot (chromium/7187) as WASMApollo\dawn.
set HERE=%~dp0
if "%DAWN_PREBUILT%"=="" set "DAWN_PREBUILT=%USERPROFILE%\WASMApollo\dawn\prebuilt"
if not exist "%DAWN_PREBUILT%\lib\webgpu_dawn.lib" (
  echo dawnrt: no Dawn at %DAWN_PREBUILT%. Run WASMApollo\dawn\build.bat once, or set DAWN_PREBUILT.
  exit /b 1
)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
if not defined VS (
  echo dawnrt: Visual Studio C++ build tools not found
  exit /b 1
)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%HERE%"
cl /nologo /std:c++17 /EHsc /O2 /W3 /DNOMINMAX /I"%DAWN_PREBUILT%\include" rt.cc ^
  /link /OUT:dawnrt.exe /LIBPATH:"%DAWN_PREBUILT%\lib" webgpu_dawn.lib winmm.lib
set RC=%errorlevel%
del /q rt.obj 2>nul
popd
if not %RC%==0 exit /b %RC%
if not exist "%HERE%webgpu_dawn.dll" copy /Y "%DAWN_PREBUILT%\bin\webgpu_dawn.dll" "%HERE%webgpu_dawn.dll" >nul
echo [dawnrt] %HERE%dawnrt.exe
exit /b 0
