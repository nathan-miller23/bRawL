FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

# Install dolpin emulator
WORKDIR /dolphin

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    qtbase5-dev qtbase5-private-dev git cmake make gcc g++ pkg-config libavcodec-dev \
    libavformat-dev libavutil-dev libswscale-dev libxi-dev libxrandr-dev libudev-dev \
    libevdev-dev libsfml-dev libminiupnpc-dev libmbedtls-dev libcurl4-openssl-dev libhidapi-dev \
    libsystemd-dev libbluetooth-dev libasound2-dev libpulse-dev libpugixml-dev libbz2-dev \
    libzstd-dev liblzo2-dev libpng-dev libusb-1.0-0-dev gettext libgtk2.0-dev

RUN git clone https://github.com/dolphin-emu/dolphin.git dolphin-emu && cd /dolphin/dolphin-emu

RUN apt-get install -y libgl1-mesa-dev libreadline-dev libxext-dev libwxgtk3.0-gtk3-dev libwxbase3.0-dev

RUN mkdir /dolphin/dolphin-emu/build
WORKDIR /dolphin/dolphin-emu/build
RUN cmake .. && make && make install

RUN rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y xdg-utils

CMD ["dolphin-emu-nogui", "--exec", "/home/Super Smash Bros Brawl.iso"]
# CMD ["dolphin-emu"]