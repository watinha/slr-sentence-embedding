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
                bibtexparser

CMD ["ash"]
