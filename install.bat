@echo off

set "requirements_txt=%~dp0\requirements.txt"
set "python_exec=..\..\..\python_embeded\python.exe"

echo Installing ComfyUI's Sound lab Nodes..

REM 调用 python_exec 并获取版本信息
for /f "tokens=*" %%i in ('%python_exec% --version') do set PYTHON_VERSION=%%i

REM 显示 Python 版本
echo %PYTHON_VERSION%

%python_exec% -c "import torch; print(torch.__version__)"

if exist "%python_exec%" (
    echo Installing with ComfyUI Portable

    @REM %python_exec% -s -m pip install --upgrade --force setuptools==69.5.1
 
    %python_exec% -s -m pip install --upgrade --force ./ffmpy-0.3.2

    %python_exec% -s -m pip install ./flash_attn-2.5.2+cu122torch2.2.0cxx11abiFALSE-cp311-cp311-win_amd64.whl
    
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