sudo yum update -y
sudo yum install -y \
    git \
    cmake \
    gcc \
    gcc-c++ \
    zlib-devel \
    bzip2 \
    bzip2-devel \
    readline-devel \
    sqlite \
    sqlite-devel \
    openssl-devel \
    xz \
    xz-devel \
    tmux

git clone https://github.com/yyuu/pyenv.git ~/.pyenv
echo '# pyenv configurations' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc

source ~/.bashrc

pyenv install -s 3.6.2  \
    && pyenv global 3.6.2 \
    && pip install --upgrade pip setuptools python-dateutil \
    && pip --no-cache-dir install -r requirements.txt
