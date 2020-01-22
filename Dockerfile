
FROM ubuntu:18.04

LABEL maintainer="hamid.arabnejad@gmail.com"

# define arguments
ARG Tutorial_dir=/home/root/turing_workshop
ARG FabSim3_dir=FabSim3
ARG FabSim3_repo=https://github.com/djgroen/FabSim3.git
ARG fdfault_dir=fdfault
ARG fdfault_repo=https://github.com/egdaub/fdfault.git
ARG mogp_emulator_dir=mogp_emulator
ARG mogp_emulator_repo=https://github.com/alan-turing-institute/mogp_emulator.git

# install dependencies needs to for FabSim3
RUN apt-get update && \
    apt-get install -y --no-install-recommends sudo git build-essential libopenmpi-dev openmpi-bin && \
    apt-get install -y --no-install-recommends openssh-server openssh-client rsync tree nano systemd && \
    apt-get install -y --no-install-recommends python3-pip python3-dev  && \
    apt-get clean autoclean && \
    apt-get autoremove --yes && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    pip3 install --upgrade pip && \
    pip install -U pip setuptools && \
    pip install pyyaml matplotlib numpy scipy pytest fabric3 cryptography 
    #cryptography==2.4.2    

# clone mogp_emulator github repository
WORKDIR ${Tutorial_dir}
RUN mkdir -p ${mogp_emulator_dir} && \
    git clone ${mogp_emulator_repo} ${mogp_emulator_dir} && \
    cd ${mogp_emulator_dir} && \
    python3 setup.py install

# clone fdfault github repository
WORKDIR ${Tutorial_dir}
RUN pwd && ls
RUN mkdir -p ${fdfault_dir} && \
    git clone ${fdfault_repo} ${fdfault_dir} && \
    cd ${fdfault_dir}/src && \
    make && \
    cd ../python && \
    python3 setup.py install

# clone FabSim3 github repository
WORKDIR ${Tutorial_dir}
RUN mkdir -p ${FabSim3_dir} && \
    git clone ${FabSim3_repo} ${FabSim3_dir}

# generate machines_user.yml file
RUN cp ${FabSim3_dir}/deploy/machines_user_example.yml ${FabSim3_dir}/deploy/machines_user.yml
RUN sed -i "s/your-username/`whoami`/g;s#~/Codes/FabSim#${Tutorial_dir}/${FabSim3_dir}#g"  ${FabSim3_dir}/deploy/machines_user.yml
RUN echo '\n\
localhost:\n\
  mpi_exec : "/usr/bin/mpiexec"\n\
  fdfault_exec : "'${Tutorial_dir}'/'${fdfault_dir}'"\n\
\n' >> ${FabSim3_dir}/deploy/machines_user.yml

# allow everyone to read and execute the file
RUN mkdir /var/run/sshd && sudo chmod -R 755 /var/run/sshd
RUN mkdir ~/.ssh && sudo chmod -R 755 ~/.ssh
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin .*$/PermitRootLogin yes/' /etc/ssh/sshd_config

#RUN fab localhost setup_fabsim
RUN rm -f ~/.ssh/id_rsa
RUN ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
RUN cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
RUN chmod og-wx ~/.ssh/authorized_keys
RUN ssh-keyscan -H localhost >> ~/.ssh/known_hosts

WORKDIR ${Tutorial_dir}/${FabSim3_dir}
RUN fab localhost install_plugin:FabDummy
RUN fab localhost install_plugin:fabmogp

# customize bashrc
RUN sed -i -e 's/#force_color_prompt=yes$/force_color_prompt=yes/'  /root/.bashrc
RUN echo 'export PS1="\[\033[01;34m\][VECMA tutorial]\[\033[01;31m\] \w\[\033[00m\] \$ "' >> /root/.bashrc

ENTRYPOINT  service ssh restart > /dev/null 2>&1 && /bin/bash
