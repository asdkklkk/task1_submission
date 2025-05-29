## 使用官方 Python 镜像
#FROM python:3.12-slim
#
## 设置工作目录
#WORKDIR /app
#
## 复制当前目录下的所有文件到工作目录
#COPY . .
#
## 安装依赖
#RUN pip install --no-cache-dir -r requirements.txt
#
## 指定运行的命令
#CMD ["python", "task1.py"]

## 使用官方 Python 镜像
#FROM python:3.12-slim
#
## 设置工作目录
#WORKDIR /app
#
## 复制当前目录下的所有文件到工作目录
#COPY . .
#
## 安装依赖
#RUN pip install --no-cache-dir -r requirements.txt
#
## 指定运行的命令
#CMD ["python", "task1.py"]

# 使用NGC Catalog中的PyTorch镜像作为基础镜像
#FROM nvcr.io/nvidia/pytorch:23.07-py3
FROM python:3.12
# 设置工作目录为/app
WORKDIR /app

# 使用构建参数设置 Hugging Face 的镜像地址
ARG HF_ENDPOINT
ENV HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

# 复制模型文件和其他必要的文件到镜像中
COPY . .


# 安装项目所需的依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 指定容器启动时运行的命令，执行task1.py脚本

CMD ["python", "/app/task1.2_int.py", "/app/val.jsonl", "/app/output"]