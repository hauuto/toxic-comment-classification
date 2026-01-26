import json
import os
import re
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import csv
from collections import Counter
from itertools import product


# Default Leet Speak mappings (letter -> list of replacements)
DEFAULT_LEET_MAP = {
    'a': ['4', '@', '/\\'],
    'b': ['8', '|3'],
    'c': ['(', '[', '<'],
    'd': ['|)', '|>'],
    'e': ['3', '€'],
    'f': ['|=', 'ph'],
    'g': ['9', '6', 'q'],
    'h': ['#', '|-|', '}{'],
    'i': ['1', '|', '!'],
    'j': ['_|'],
    'k': ['|<', '|{'],
    'l': ['1', '|_', '|'],
    'm': ['/\\/\\', '|\\/|'],
    'n': ['/\\/', '|\\|'],
    'o': ['0', '()'],
    'p': ['|>', '|*'],
    'q': ['9', '0_'],
    'r': ['|2', '|?', '12'],
    's': ['5', '$'],
    't': ['7', '+'],
    'u': ['|_|', '\\_/', 'v'],
    'v': ['\\/', '>'],
    'w': ['\\/\\/', '|/\\|', 'vv'],
    'x': ['><', '}{'],
    'y': ['`/', '¥'],
    'z': ['2', '7_'],
    # Vietnamese specific (shortened)
    'đ': ['d', 'dd', 'dj'],
}

class VietnameseDictionary:
    _instance = None
    _words = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        pass
