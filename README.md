This is our README, which maybe we will fill out at some point

I'm going to commit this as a test

USING DOCKER:
install docker using brew or installer of ur choice.
Install Xquartz on local machine
Get docker desktop and launch to start daemon  
login to docker from terminal with docker login -u <username>
run docker build -t image-name . in terminal
If build is failing for same reasons after making changes run with --no-cache
run xhost + 127.0.0.1 in terminal
start xquartz server from applications/utililites/xquartz
check box for allowing network connections under xquartz settings
restart xquartz
then command
docker run -it --rm -e DISPLAY=host.docker.internal:0 -e LIBGL_ALWAYS_SOFTWARE=1 image name

