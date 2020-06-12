FROM debian:latest
MAINTAINER Conda Development Team <conda@continuum.io>

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

# Add and install Python modules
#ADD requirements.txt /src/requirements.txt
#RUN cd /src; pip install -r requirements.txt

RUN conda install pandas
RUN pip install streamlit
RUN conda install -c conda-forge pydeck

# Bundle app source
ADD src/* src/
add data/* data/

# Expose
EXPOSE  8501

# Run
CMD ["streamlit", "run", "src/windsite.py"]