import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys

ROOT=Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0,str(ROOT))

from src.llm import LoaclLLM
from src.utils import (
    Benchmark,
    BENCHMARKS,
    load_tokenizer,
    load_benchmark,
)
from src.prompts import (
    prompt4whether_retrieval,
    prompt4what_aspect,
    prompt4what_choice,
    construct_text4rerankng,
)
from src.config import WWW27Config
from src.aspects import Aspect, ASPECTS, AspectSelector




def main():
    pass



if __name__ == '__main__':
    main()









