#!/bin/bash
if [[ $(echo $USER) == *"mcoe"* ]]
	then
		echo "exporting proxy variables..."
		export http_proxy=http://192.168.0.2:3128/
		export https_proxy=https://192.168.0.2:3128/
	else
		echo "Not at College."
fi
echo "Checking if openjdk exists..."
if [[ $(java -version 2>&1) == *"OpenJDK"* ]];
	then
		echo "Removing openjdk..."
		sudo apt-get purge openjdk-\*;
	else
		echo "No openjdk found, skipping openjdk uninstallation.";
fi
echo "Creating dir... /usr/local/java/"
sudo mkdir -p /usr/local/java/
echo "Copying jdk-8u101-linux-i586 -> /usr/local/java/"
sudo cp -R /media/$USER/AJ/jdk-8u101-linux-i586.tar.gz /usr/local/java/
cd /usr/local/java/
sudo tar -xvf jdk-8u101-linux-i586.tar.gz
echo "setting path variables..."
export JAVA_HOME=/usr/local/java/jdk1.8.0_101
export JRE_HOME=$JAVA_HOME/jre
export PATH=$PATH:$JAVA_HOME/bin:$JRE_HOME/bin

sudo -E update-alternatives --install "/usr/bin/java" "java" "/usr/local/java/jdk1.8.0_101/jre/bin/java" 1
sudo -E update-alternatives --install "/usr/bin/javaws" "java" "/usr/local/java/jdk1.8.0_101/jre/bin/javaws" 1
sudo -E update-alternatives --install "/usr/bin/javac" "javac" "/usr/local/java/jdk1.8.0_101/bin/javac" 1
sudo -E update-alternatives --set java /usr/local/java/jdk1.8.0_101/jre/bin/java
sudo -E update-alternatives --set javaws /usr/local/java/jdk1.8.0_101/jre/bin/javaws
sudo -E update-alternatives --set javac /usr/local/java/jdk1.8.0_101/bin/javac
echo "Installation Complete."
java -version
