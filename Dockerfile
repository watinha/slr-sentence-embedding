FROM python:alpine

RUN apk add alpine-sdk \
            cython \
            openblas-dev \
            zlib-dev
RUN pip install numpy \
                pandas \
                sklearn \
                imblearn \
                openpyxl \
                bibtexparser \
                nltk \
                np

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4

CMD ["ash"]
