#!/bin/bash
if [[ $(echo $USER) == *"mcoe"* ]]
	then
		echo "exporting proxy variables..."
		export http_proxy=http://192.168.0.2:3128/
		export https_proxy=https://192.168.0.2:3128/
	else
		echo "Not at College."
fi
sudo apt-get update && apt-get -y install screen hexchat
cd /opt/
wget -c http://sourceforge.net/projects/xdman/files/xdm-jre-32bit.tar.xz
tar -xvf xdm-jre-32bit.tar.xz
ln -s /usr/bin/xdman /opt/xdm/xdm
