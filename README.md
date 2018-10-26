# Get started

## Why

The goal of the project is to explore machine learning using python and some of the popular machine learning libraries. The project takes inspiration from a machine leanring course taught using matlab, so the assigments are based on the course itself and translated into python. The results are collected and verfied against the matlab counterpart

## Create image from Dockerfile
`
docker build -t ${"image name"} . 
`

## Create container from image
`
docker run -it -p 8888:8888 -v ${your local directory}:${directory on the VM} --name ${container name} ${image name} / bin/bash
`

## Start jupyter notebook
`
jupyter notebook --no-browser --allow-root --ip 0.0.0.0 --port 8888 ${directory on the VM}
`

🤗 Happy developing 🤗