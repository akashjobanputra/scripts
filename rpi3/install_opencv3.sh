#/bin/bash
mkdir opencv_3
cd opencv_3
echo "Downloading installation files..."
git clone https://gist.github.com/8dc0c4d7a96ffc72ac7885123f53f7e6.git
cd 8dc0c4d7a96ffc72ac7885123f53f7e6/
echo "Installating prerequisites..."
sudo apt install $(cat cv2_requirements_apt.txt)
sudo ldconfig
echo "Installing opencv_3..."
sudo dpkg -i opencv_3.1.0_armhf.deb