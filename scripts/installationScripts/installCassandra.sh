#!/bin/bash
#Script for installation of Cassandra, check version of JDK if not OracleJDK then installs it and continues the installation of cassandra
if [[ $(java -version 2>&1) == *"OpenJDK"* ]];
	then 
		echo 'Purging OpenJDK';
		sudo apt-get purge -y openjdk-\*
		sudo apt-get autoremove
		echo 'OpenJDK removed, installing Oracle JDK';
		tar -xf jdk-8u101-linux-i586.tar.gz
		mkdir /usr/lib/jvm/
		echo 'sudo mv /path/to/jdk1.8.0_101 /usr/lib/jvm/oracle_jdk8'
		sudo mv /path/to/jdk1.8.0_101 /usr/lib/jvm/oracle_jdk8
		sudo update-alternatives --install /usr/bin/java java /usr/lib/jvm/oracle_jdk8/jre/bin/java 2000
		sudo update-alternatives --install /usr/bin/javac javac /usr/lib/jvm/oracle_jdk8/bin/javac 2000
		sudo update-alternatives --install /usr/bin/javaws javaws /usr/lib/jvm/oracle_jdk8/jre/bin/javaws 2000
		echo 'JAVA_HOME=/usr/lib/jvm/oracle_jdk8' >> /etc/profile
		echo 'JRE_HOME=$JAVA_HOME/jre' >> /etc/profile
		echo 'PATH=$PATH:$JAVA_HOME/bin:$JRE_HOME/bin' >> /etc/profile
		echo 'export JAVA_HOME' >> /etc/profile
		echo 'export JRE_HOME' >> /etc/profile
		echo 'export PATH' >> /etc/profile
		. /etc/profile
		echo $JAVA_HOME
		echo $JRE_HOME
		echo 'Installation of Oracle JDK8 complete, now proceeding for installation of cassandra.';
	else
		echo 'Oracle JDK it seems, proceeding to installation of cassandra';
fi
echo "deb http://www.apache.org/dist/cassandra/debian 22x main" | sudo tee -a /etc/apt/sources.list.d/cassandra.sources.list
echo "deb-src http://www.apache.org/dist/cassandra/debian 22x main" | sudo tee -a /etc/apt/sources.list.d/cassandra.sources.list
echo 'Adding installation keys...'
	gpg --keyserver pgp.mit.edu --recv-keys F758CE318D77295D
	gpg --export --armor F758CE318D77295D | sudo apt-key add -
	gpg --keyserver pgp.mit.edu --recv-keys 2B5C1B00
	gpg --export --armor 2B5C1B00 | sudo apt-key add -
	gpg --keyserver pgp.mit.edu --recv-keys 0353B12C
	gpg --export --armor 0353B12C | sudo apt-key add -
echo 'sudo apt update...executing'
	sudo apt update
echo 'Finally installing cassandra'
	sudo apt -y install cassandra
echo 'Done, I guess! :)'