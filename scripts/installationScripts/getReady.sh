#!/bin/bash
export http_proxy=http://192.168.0.2:3128/
export https_proxy=https://192.168.0.2:3128/
sudo apt-get update && apt-get -y install screen hexchat unity-tweak-tool sysvbanner cmatrix
