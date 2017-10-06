# Facial and Pupil Tracking

Hello, this is a starter project for myself to be familiar with computer vision. 

This means that this README is written with only my future self in thought and instructions and descriptions are short. 

For anyone reading this, you may use this for inspiration. 

## Getting Started

Assuming a new macbook and installed homebrew...This information is copy pasted from http://www.cs.sjsu.edu/~bruce/fall_2016_cs_160_handout_how_to_build_and_install_openface_on_linux_mint.html

OBJECTIVE: to compile and install the Openface face detection library on Linux Mint.

Warning: This is a going to be a big step-by-step. There are lots of dependencies on specific libraries. We begin the journey...

Launch a terminal window (also known as a command line prompt or shell).
Step 1 of 39: In terminal window, log in as root
In your terminal window type the following at the command prompt: su
You will be prompted to enter the root password. Enter that password.
Your terminal window should now display a "#" (hash mark) prompt. This indicates you are logged in as root!
Step 2 of 39: In terminal window, type the following command as root
First we need to update our cache of software libraries that are available to install or update.
In your terminal window type the following at the command prompt (you should be root): apt-get update
Step 3 of 39: Install dependency libraries and tools
Now that we've updated our cache, the next step is to install a series of development libraries and tools.
In your terminal window type the following at the command prompt (you should be root): apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev libboost-all-dev libtbb-dev libopenblas-dev libeigen3-dev default-jdk ant
Step 4 of 39: Log out as root
Now that we've installed all these development libraries and tools, we can log out as the root user. It's always better to run as a non-priviledged user whenever possible.
In your terminal window type the following at the command prompt (you should be root): exit
Step 5 of 39: Change to Downloads directory
You should still be logged into your terminal window but as a non-priviledged user (i.e. non-root).
Your command prompt should no longer display a "#" (hash mark) prompt.
In your terminal window type the following at the command prompt: cd ~/Downloads/
Step 6 of 39: download Opencv version 3.1.0
In your terminal window type the following at the command prompt: wget https://github.com/Itseez/opencv/archive/3.1.0.zip -O opencv-3.1.0.zip
Step 7 of 39: unzip Opencv archive
In your terminal window type the following at the command prompt: unzip opencv-3.1.0.zip
Step 8 of 39: go inside the Opencv directory
In your terminal window type the following at the command prompt: cd opencv-3.1.0
Step 9 of 39: make a build directory
Now that you are inside the opencv-2.4.11 source tree, we need to make a new build directory.
In your terminal window type the following at the command prompt: mkdir build
Step 10 of 39: go inside the build directory
In your terminal window type the following at the command prompt: cd build
Step 11 of 39: create a makefile for building opencv
In your terminal window type the following at the command prompt: cmake -D WITH_TBB=ON ..
Step 12 of 39: build Opencv version 3.1.0 from source
Now that makefiles have been created, it is time to compile Opencv from source!
Compiling can take a long time. I suggest using the -j4 option (below) to specify four cores for compiling concurrently. It will build faster than single core (default)!
In your terminal window type the following at the command prompt: make -j4
Step 13 of 39: log in as root user
Now that OpenCV has been successfully compiled, we need to log in as root once again to install the binaries and libraries.
In your terminal window type the following at the command prompt: su
Step 14 of 39: install Opencv version 2.4.11
Your terminal window should now display a "#" (hash mark) prompt. This indicates you are logged in as root!
In your terminal window type the following at the command prompt (you should be root): make install
Step 15 of 39: logout of terminal window as user root
Now it's time to log out as root user in the terminal window.
We still have more work to do but we should run as a non-priviledged user once again.
In your terminal window type the following at the command prompt (you should be root): exit
Step 16 of 39: change to the Documents directory
You should still be logged into your terminal window but as a non-priviledged user (i.e. non-root).
Your command prompt should no longer display a "#" (hash mark) prompt.
In your terminal window type the following at the command prompt: cd ~/Downloads/
Step 17 of 39: Download dlib source via wget
In your terminal window type the following at the command prompt: wget http://dlib.net/files/dlib-19.1.tar.bz2
Step 18 of 39: uncompress dlib source
In your terminal window type the following at the command prompt: bunzip2 dlib-19.1.tar.bz2
Step 19 of 39: untar dlib source
In your terminal window type the following at the command prompt: tar xf dlib-19.1.tar
Step 20 of 39: enter the dlib source directory
In your terminal window type the following at the command prompt: cd dlib-19.1
Step 21 of 39: enter the dlib source directory
In your terminal window type the following at the command prompt: cd dlib-19.1
Step 22 of 39: create a build directory under dlib source directory
In your terminal window type the following at the command prompt: mkdir build
Step 23 of 39: enter the build directory under dlib source directory
In your terminal window type the following at the command prompt: cd build
Step 24 of 39: create a makefile for dlib
In your terminal window type the following at the command prompt: cmake ..
Step 25 of 39: build dlib from source
Now that makefiles have been created, it is time to compile dlib from source!
Compiling can take a long time. I suggest using the -j4 option (below) to specify four cores for compiling concurrently. It will build faster than single core (default)!
In your terminal window type the following at the command prompt: make -j4
Step 26 of 39: log in as root user
Now that OpenCV has been successfully compiled, we need to log in as root once again to install the binaries and libraries.
In your terminal window type the following at the command prompt: su
Step 27 of 39: install Opencv version 2.4.11
Your terminal window should now display a "#" (hash mark) prompt. This indicates you are logged in as root!
In your terminal window type the following at the command prompt (you should be root): make install
Step 28 of 39: logout of terminal window as user root
Now it's time to log out as root user in the terminal window.
We still have more work to do but we should run as a non-priviledged user once again.
In your terminal window type the following at the command prompt (you should be root): exit
Step 29 of 39: change to the Documents directory
You should still be logged into your terminal window but as a non-priviledged user (i.e. non-root).
Your command prompt should no longer display a "#" (hash mark) prompt.
In your terminal window type the following at the command prompt: cd ~/Downloads/
Step 30 of 39: Download OpenFace via git
In your terminal window type the following at the command prompt: git clone https://github.com/TadasBaltrusaitis/OpenFace.git
Step 31 of 39: change directory into OpenFace source
In the previous step you downloaded the source tree for OpenFace from GitHub. Now change into the Openface source directory.
In your terminal window type the following at the command prompt: cd OpenFace/
Step 32 of 39: create a build directory under OpenFace source tree
In your terminal window type the following at the command prompt: mkdir build
Step 33 of 39: go to the build directory under OpenFace source tree
In your terminal window type the following at the command prompt: cd build
Step 34 of 39: create a makefile for OpenFace
In your terminal window type the following at the command prompt: cmake ..
Step 35 of 39: build OpenFace from source
Now that makefiles have been created, it is time to compile OpenFace from source!
Compiling can take a long time. I suggest using the -j4 option (below) to specify four cores for compiling concurrently. It will build faster than single core (default)!
In your terminal window type the following at the command prompt: make -j4
Step 36 of 39: log in as root user
Now that OpenCV has been successfully compiled, we need to log in as root once again to install the binaries and libraries.
In your terminal window type the following at the command prompt: su
Step 37 of 39: install OpenFace
Your terminal window should now display a "#" (hash mark) prompt. This indicates you are logged in as root!
In your terminal window type the following at the command prompt (you should be root): make install
Step 38 of 39: logout of terminal window as user root
Now it's time to log out as root user in the terminal window.
In your terminal window type the following at the command prompt (you should be root): exit
Step 39 of 39: close terminal window (we are all done)
In your terminal window type the following at the command prompt (as a non-priviledged user): exit
Your terminal window should now be closed.

### Prerequisites

What things you need to install the software and how to install them

CMake, OpenCV, OpenFace

## Running the tests

Run executable

### And coding style tests

Create a new selfie video.

Use FFMPEG to split the images, and then run 68 points.

## Built With

http://thume.ca/projects/2012/11/04/simple-accurate-eye-center-tracking-in-opencv/

## Acknowledgments

* Hat tip to anyone who tries to use this because this is casually written README
* Inspiration: Professor Bruce
