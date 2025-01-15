FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Install basic utilities
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Initialize conda in bash
RUN conda init bash

# Install NVIDIA Container Toolkit
RUN apt-get update && apt-get install -y \
    nvidia-container-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install zsh and required packages
RUN apt-get update && apt-get install -y \
    zsh \
    git-core \
    curl \
    fonts-powerline \
    && rm -rf /var/lib/apt/lists/*

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Install useful zsh plugins
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Configure zsh
RUN sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc

# Set zsh as default shell
RUN chsh -s $(which zsh)
