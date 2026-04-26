import sys
import os

# Make webapp/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'webapp'))

from app import app
