@echo off

set "requirements_txt=%~dp0\requirements.txt"
set "python_exec=..\..\..\python_embeded\python.exe"

echo Installing ComfyUI's Sound lab Nodes..

if exist "%python_exec%" (
    echo Installing with ComfyUI Portable

    @REM %python_exec% -s -m pip install --upgrade --force setuptools==69.5.1
 
    %python_exec% -s -m pip install --upgrade --force ./ffmpy-0.3.2

    for /f "delims=" %%i in (%requirements_txt%) do (
        %python_exec% -s -m pip install "%%i" -i https://pypi.tuna.tsinghua.edu.cn/simple
    )
) else (
    echo Installing with system Python
    for /f "delims=" %%i in (%requirements_txt%) do (
        pip install "%%i" -i https://pypi.tuna.tsinghua.edu.cn/simple
    )
)

pause