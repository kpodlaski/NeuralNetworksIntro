FROM nvcr.io/nvidia/pytorch:25.09-py3
LABEL authors="kp"
ENV PIP_ROOT_USER_ACTION=ignore
RUN python3 -m pip install --upgrade pip
RUN pip3 install torchsummary
RUN pip3 install --upgrade certifi
RUN pip3 install pip-system-certs

ENTRYPOINT ["top", "-b"]
