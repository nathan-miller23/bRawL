FROM ubuntu:18.04

RUN apt update && apt install -y python3-pip

RUN apt update && apt install -y build-essential cmake libevdev-dev libudev-dev libgl1-mesa-dev \
                                 libusb-1.0.0-dev libao-dev libpulse-dev libxrandr-dev libopenal-dev \
                                 libasound2-dev libzmq3-dev libgtk2.0-dev libpng-dev qtbase5-dev qtbase5-private-dev
RUN apt install -y git

WORKDIR /ssbm
RUN git clone https://github.com/Gurvan/ssbm_gym.git 
RUN pip3 install -e ./ssbm_gym

WORKDIR /dolphin

RUN git clone --depth 1 https://github.com/Gurvan/dolphin.git dolphin_src

RUN mkdir /dolphin/dolphin_src/build
WORKDIR /dolphin/dolphin_src/build

RUN cmake .. -DENABLE_HEADLESS=false -DENABLE_NOGUI=false
RUN make -j 2
RUN cp -R ../Data/Sys Binaries/

RUN mkdir /dolphin/dolphin_src/build_nogui
WORKDIR /dolphin/dolphin_src/build_nogui

RUN cmake ..
RUN make -j 2
RUN cp -R ../Data/Sys Binaries/

RUN ln -s /dolphin/dolphin_src/build_nogui/Binaries/dolphin-emu-nogui /ssbm/ssbm_gym/dolphin-exe/
RUN ln -s /dolphin/dolphin_src/build/Binaries/dolphin-emu /ssbm/ssbm_gym/dolphin-exe/

# CMD ["python3", "-u", "/ssbm/ssbm_gym/test_env.py"]

COPY ./exec.sh ./exec.sh
