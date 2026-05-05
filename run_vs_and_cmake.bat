@echo off
setlocal
echo Removing Strawberry entries from PATH for this session (if present)
set "PATH=%PATH:C:\Strawberry\c\bin;=%"
set "PATH=%PATH:;C:\Strawberry\c\bin=%"
echo PATH after filter:
echo %PATH%

call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
where cl
where nvcc
where cmake

if exist build_cuda_verify (
	echo Removing existing build_cuda_verify directory
	rmdir /s /q build_cuda_verify
)

cmake -S . -B build_cuda_verify -DNN_ENABLE_LTO=OFF
echo cmake exit code %ERRORLEVEL%
endlocal
