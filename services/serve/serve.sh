#!/bin/sh

python -m mlc_llm compile Llama-3.1-8B-Instruct-q4f16_1-MLC/ \
  --opt O3 \
  --overrides "disaggregation=1" \
  -o Llama-3.1-8B-Instruct-q4f16_1-MLC/lib_disagg.so

python serve.py