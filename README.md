## GPU

### Windows
    Version             Python version	Compiler	Build tools     cuDNN       CUDA
    tensorflow-2.4.0    3.6-3.8	        GCC 7.3.1	Bazel 3.1.0     8.0         11.0

## SUPPORT INSTALLATION
    pip install tf-nightly

## GAN TUTORIAL
- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

## TRANSFORMER INSTALL FROM SOURCE

### v1.4.4
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install -e .

## NLP
    pip install spacy-stanza==0.2.1
    python -m spacy download es_core_news_sm
    (other packages: es_core_news_md and en_core_web_md)

## TEXTRACT
    pip install amazon-textract-prettyprinter

## PYTORCH INSTALLATION FOR CUDA 11

### Windows
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
