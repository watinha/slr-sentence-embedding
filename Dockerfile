FROM python:3.8.16-slim-bullseye

RUN apt-get update
RUN apt-get -y install build-essential \
                       gfortran

RUN pip install numpy \
                pandas \
                sklearn \
                imblearn \
                openpyxl \
                bibtexparser \
                nltk \
                np \
                cython \
                gensim

RUN python -m nltk.downloader punkt \
                              stopwords \
                              averaged_perceptron_tagger \
                              wordnet \
                              omw-1.4

CMD ["bash"]
