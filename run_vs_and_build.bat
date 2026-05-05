@echo off
setlocal
echo Removing Strawberry entries from PATH for this session (if present)
set "PATH=%PATH:C:\Strawberry\c\bin;=%"
set "PATH=%PATH:;C:\Strawberry\c\bin=%"
echo PATH after filter:
echo %PATH%

call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
cmake --build build_cuda_verify --config Release --parallel
echo build exit code %ERRORLEVEL%
endlocal
