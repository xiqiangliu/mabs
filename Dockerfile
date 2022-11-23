FROM ucsdets/datahub-base-notebook:2022.3-stable

# RUN apt-get update && apt-get install -y nano && rm -rf /var/lib/apt/lists/*
# RUN adduser --disabled-password --gecos '' mabs
# RUN touch /usr/local/bin/start-notebook.sh && chmod a+x /usr/local/bin/start-notebook.sh

USER jovyan
COPY --chown=1000 . /home/jovyan/mabs
RUN pip install --upgrade pip wheel && pip cache purge
RUN pip install --upgrade -e mabs/ && pip uninstall -y mabs && pip cache purge && rm -r mabs/

CMD ["/bin/bash"]
