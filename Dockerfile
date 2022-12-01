FROM ucsdets/datahub-base-notebook:2022.3-stable

USER jovyan
COPY --chown=1000 . /home/jovyan/mabs
RUN pip install --upgrade pip wheel && pip cache purge
RUN pip install --upgrade -e "mabs/[jit]" && pip cache purge

CMD ["/bin/bash"]
