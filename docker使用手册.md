# 创建docker的流程
### 1、下载CUDA
ubuntu版本选20.04，版本不影响项目运行
https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local
![image](https://github.com/user-attachments/assets/b3123201-44dd-404a-823d-c99fa3621596)

export PATH=$PATH:/usr/local/cuda-11.3/bin

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64

### 2、下载anaconda
这里需要先把wget下载好
超详细Ubuntu安装Anaconda步骤+Anconda常用命令    https://blog.csdn.net/KRISNAT/article/details/124041869

### 3、用conda创建虚拟环境，之后只需进入环境后cd到目标文件夹即可，无需配置解释器。

###4、配置ｐｙｔｒｏｃｈ环境

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113


# 创建项目的流程
### 1、使用docker时，首先需要选中一个镜像image，来创建一个容器（docker run -it --name=”自定义”）
docker run -it --gpus all --name ythceshi2 ubuntu（这段指令用来创建一个基于Ubuntu的可以使用gpu的名为ythceshi2的docker）

### 2、可以使用dockerfile快速配置环境（可以不用）

### 3、git需要下载，使用git命令从github抓取项目（例如git clone https://github.com/VITA-Group/TransGAN.git）
git下载执行以下两行代码：
apt update
apt install git

### 4、确保 Python 和 Pip 已安装（python3 --version、pip3 --version；如果没安装就sudo apt update、sudo apt install python3 python3-pip python3-venv）

### 5、创建虚拟环境
1.cd到项目路径（例如cd TransGAN）

2.在项目路径下创建虚拟环境 python3 -m venv 环境名（例如python3 -m venv TransGAN）

3.激活虚拟环境（例如source TransGAN/bin/activate）

### 6、配置解释器
1.确保项目文件夹已在 VS Code 中打开： 你的 VS Code 窗口应该已经连接到你的 Linux 服务器，并且打开了你的项目文件夹。

2.打开命令面板： 按 Ctrl+Shift+P (或者 F1)。

3.搜索命令： 输入 Python: Select Interpreter 并选择这个命令。

4.选择虚拟环境： VS Code 会显示一个列表，列出它在你的服务器上检测到的所有 Python 环境。（一般在虚拟环境文件夹的bin里面的一个python文件）





# 常用指令
nvidia-smi：查看显卡情况

pip list：查看当前环境中已安装的包

pwd：输出你当前所在的完整绝对路径

ls：：查看目录内容

ls -l：更详细的目录内容

ls -a：查看所有文件，包括隐藏文件（.开头的文件）


# 一些网址
Github 生成SSH秘钥（详细教程）  https://blog.csdn.net/qq_35495339/article/details/92847819

Docker命令大全        https://www.runoob.com/docker/docker-run-command.html

Docker使用教程        https://www.cnblogs.com/zha0gongz1/p/12227485.html

vscode通过ssh连接远程服务器+免密登录（图文）  https://blog.csdn.net/savet/article/details/131683156

如何在Ubuntu 20.04安装wget命令      https://www.myfreax.com/how-to-install-wget-command-on-ubuntu-20-04/

Ubuntu20.04系统配置Pytorch环境(GPU版)      https://blog.csdn.net/m0_55127902/article/details/135677560

（已解决）vscode终端的虚拟环境显示两个环境名    https://blog.csdn.net/weixin_57253447/article/details/145852724




