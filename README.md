# Get started

## Create image from Dockerfile
run the following line:
```
docker build -t ${"image name"} . 
```

## Create container from image
run the following line:
```
docker run -it -p 8888:8888 -v ${your local directory}:${directory on the VM} --name ${container name} ${image name} /bin/bash
```

## Start jupyter notebook
run the following line:
```
jupyter notebook --no-browser --allow-root --ip 0.0.0.0 --port 8888 ${directory on the VM}
```