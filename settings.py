"""
Project configuration settings
"""

# Hugging Face API settings
HUGGINGFACE_API_URL = "https://huggingface.co"
HUGGINGFACE_TIMEOUT = 30  # seconds

# Lilypad network settings
LILYPAD_NETWORK = {
    'testnet': {
        'url': 'https://testnet.lilypad.tech',
        'arbitrum_rpc': 'https://sepolia-rollup.arbitrum.io/rpc'
    },
    'mainnet': {
        'url': 'https://lilypad.tech',
        'arbitrum_rpc': 'https://arb1.arbitrum.io/rpc'
    }
}

# Docker settings
DOCKER_REGISTRY = "dockerhub.io"
DOCKER_IMAGE_PREFIX = "lilypad"

# Model validation settings
MAX_MODEL_SIZE = 50 * 1024 * 1024 * 1024  # 50GB max model size
MIN_GPU_MEMORY = 8  # GB
SUPPORTED_FRAMEWORKS = ['pytorch', 'tensorflow']
SUPPORTED_TASKS = [
    'text-classification',
    'text-generation',
    'translation',
    'summarization',
    'question-answering',
    'image-classification',
    'object-detection',
    'semantic-segmentation',
    'text-to-image',
    'text-to-speech',
    'automatic-speech-recognition',
    'video-classification',
    'visual-question-answering',
    'point-cloud',
    'graph-processing'
]

# Input/Output settings
MAX_BATCH_SIZE = 32
MAX_TEXT_LENGTH = 8192
MAX_IMAGE_SIZE = (2048, 2048)
MAX_AUDIO_LENGTH = 600  # seconds
MAX_VIDEO_LENGTH = 300  # seconds

# Resource requirements
DEFAULT_CPU_COUNT = 1000  # millicpus
DEFAULT_RAM = 8000  # MB
DEFAULT_GPU_COUNT = 1
DEFAULT_TIMEOUT = 1800  # seconds

# Development settings
DEBUG = True
TEMPLATE_AUTO_RELOAD = True