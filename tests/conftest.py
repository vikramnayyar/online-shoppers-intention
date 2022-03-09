import pytest
import yaml
import os
import json

@pytest.fixture
def config(config_path = "params.yaml"):
    
