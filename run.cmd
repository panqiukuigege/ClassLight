@echo off
chcp 65001
echo 本次制作组室来自于山东省青岛市第五中学2025级3班的郑子浩、陈思达、陈思烨、徐世家、李承锦、胡皓淞
echo 此程序已开源到Github，链接https://github.com/panqiukuigege/ClassLight
echo 这个开源项目须自己接入智能灯或者智能开关
echo 需要安装Python才能运行此程序，否则会报错
echo 本开源项目遵循GPL-3.0开源协议
echo [检测] 正在检测Python环境，请安装python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] Python未安装，请先安装Python喵...   
    echo.
    echo 按任意键退出...
    pause >nul
    exit /b 1
)
echo[安装]正在安装opencv-python库（如安装则会跳过安装）
pip install opencv-python -i https://mirrors.aliyun.com/pypi/simple/
echo[安装]正在安装numpy库（如安装则会跳过安装）
pip install numpy -i https://mirrors.aliyun.com/pypi/simple/
python code.py