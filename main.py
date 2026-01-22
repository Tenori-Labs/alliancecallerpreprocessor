from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
from io import BytesIO
from pathlib import Path
import uuid
import tarfile
import shutil
import threading
import time
import atexit
from datetime import datetime
from enum import Enum
from urllib.parse import urlparse, parse_qs, unquote

try:
    from pyngrok import ngrok
    PYNGROK_AVAILABLE = True
except ImportError:
    PYNGROK_AVAILABLE = False
    print("⚠️  pyngrok not found. Install with: pip install pyngrok")

try:
    import fitz  # PyMuPDF..
except ImportError:
    print("PyMuPDF not found. Please install it using:")
    print("  pip install pymupdf")
    sys.exit(1)

try:
    from PIL import Image
    import pytesseract
except ImportError:
    print("OCR dependencies not found. Please install them using:")
    print("  pip install pillow pytesseract")
    sys.exit(1)

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not found. Please install it using:")
    print("  pip install chromadb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not found. Installing automatically...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
        print("✓ sentence-transformers installed successfully")
        from sentence_transformers import SentenceTransformer
    except subprocess.CalledProcessError:
        print("✗ Failed to install sentence-transformers automatically.")
        print("Please install it manually using: pip install sentence-transformers")
        sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("numpy not found. Please install it using:")
    print("  pip install numpy")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI not found. Please install it using:")
    print("  pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not found. Please install it using:")
    print("  pip install python-dotenv")
    sys.exit(1)

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("boto3 not found. Please install it using:")
    print("  pip install boto3")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("requests not found. Please install it using:")
    print("  pip install requests")
    sys.exit(1)

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
except ImportError:
    print("reportlab not found. Please install it using:")
    print("  pip install reportlab")
    sys.exit(1)


try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    from bson import ObjectId
    MONGODB_AVAILABLE = True
except ImportError:
    print("pymongo not found. Please install it using:")
    print("  pip install pymongo")
    MONGODB_AVAILABLE = False


try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    import pickle
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    print("Google Calendar libraries not found. Please install them using:")
    print("  pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
    GOOGLE_CALENDAR_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# Configure Tesseract path for Windows
TESSERACT_FOUND = False
if sys.platform == 'win32':
    import shutil
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        TESSERACT_FOUND = True
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        ]
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                TESSERACT_FOUND = True
                break
else:
    TESSERACT_FOUND = True

# Initialize FastAPI.
app = FastAPI(
    title="PDF Text Extraction & Query API with WhatsApp",
    description="Extract text from PDFs, store in ChromaDB, query with GPT-4o, and send via WhatsApp",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job tracking for async extraction
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# In-memory job storage (could be replaced with Redis/MongoDB for production)
extraction_jobs = {}
job_lock = threading.Lock()
backup_lock = threading.Lock()  # Lock to prevent concurrent ChromaDB backups
extraction_in_progress = False  # Flag to track if extraction is in progress
extraction_lock = threading.Lock()  # Lock to protect extraction_in_progress flag

# Cache for unique PDF names to avoid expensive collection.get() calls
_cached_pdf_names = set()
_pdf_names_cache_lock = threading.Lock()
_pdf_names_cache_valid = False  # Flag to indicate if cache needs refresh

# Ngrok tunnel variable
ngrok_tunnel = None


def start_ngrok_tunnel(port=8080):
    """Start ngrok tunnel for the FastAPI service"""
    global ngrok_tunnel
    
    if not PYNGROK_AVAILABLE:
        print("⚠️  pyngrok not available, skipping ngrok tunnel")
        return None
    
    ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
    
    if ngrok_auth_token:
        try:
            ngrok.set_auth_token(ngrok_auth_token)
            print("✓ Using ngrok auth token from environment")
        except Exception as e:
            print(f"⚠️  Failed to set ngrok auth token: {e}")
    else:
        print("⚠️  NGROK_AUTH_TOKEN not set, using free ngrok (may have limitations)")
    
    try:
        ngrok_tunnel = ngrok.connect(port, "http")
        public_url = ngrok_tunnel.public_url
        
        print("\n" + "="*80)
        print("🚀 ngrok tunnel started successfully!")
        print(f"📞 Public URL: {public_url}")
        print(f"🌐 Extract endpoint: {public_url}/extract")
        print(f"🔍 Query endpoint: {public_url}/query")
        print(f"💚 Health check: {public_url}/health")
        print("="*80 + "\n")
        
        atexit.register(cleanup_ngrok)
        return public_url
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️  Failed to start ngrok tunnel: {error_msg}")
        
        # Check if it's a version error
        if "too old" in error_msg.lower() or "minimum" in error_msg.lower():
            print("="*80)
            print("❌ ngrok agent version is too old!")
            print("💡 Solution: Update pyngrok to the latest version:")
            print("   pip install --upgrade pyngrok")
            print("="*80)
        
        return None


def cleanup_ngrok():
    """Clean up ngrok tunnel on exit"""
    global ngrok_tunnel
    if ngrok_tunnel:
        try:
            ngrok.disconnect(ngrok_tunnel.public_url)
            ngrok.kill()
            print("✓ ngrok tunnel closed")
        except Exception as e:
            print(f"⚠️  Error closing ngrok tunnel: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up ngrok tunnel on shutdown"""
    cleanup_ngrok()


@app.head("/health")
async def health_check():
    """
    Health check endpoint (HEAD request).
    Returns 200 OK if the service is running.
    """
    return None


# Google Calendar configuration
CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar']

# Initialize OpenAI client (with error handling)
# Note: OpenAI client is used for other features (like calendar, course determination), not for embeddings
# Embeddings are handled by SentenceTransformer (lazy initialization via ensure_models_initialized())
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("\n⚠️  WARNING: OPENAI_API_KEY not found in .env file or environment!")
    print("Some features may not work without an API key.")
    openai_client = None
else:
    openai_client = OpenAI(api_key=openai_api_key)

# Initialize S3 client (with error handling) - MUST be before ChromaDB restore
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "alliancewidget")

def get_bucket_region(bucket_name, access_key, secret_key, default_region):
    """Get the actual region of an S3 bucket."""
    try:
        # Create a temporary client with default region to get bucket location
        temp_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=default_region
        )
        # Get bucket location (returns 'us-east-1' as None or the actual region)
        response = temp_client.get_bucket_location(Bucket=bucket_name)
        location = response.get('LocationConstraint')
        # If location is None or empty, it means us-east-1
        if location is None or location == '':
            return 'ap-south-1'
        return location
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        # If bucket doesn't exist or access denied, return default
        if error_code in ['NoSuchBucket', 'AccessDenied', '403']:
            print(f"  ⚠ Could not detect bucket region ({error_code}), using configured: {default_region}")
            return default_region
        # For other errors, try common regions
        print(f"  ⚠ Error detecting bucket region: {e}")
        return default_region
    except Exception as e:
        print(f"  ⚠ Could not detect bucket region: {e}, using configured: {default_region}")
        return default_region

if not aws_access_key or not aws_secret_key:
    print("\n⚠️  WARNING: AWS credentials not found in .env file or environment!")
    print("The S3 upload functionality will not work without AWS credentials.")
    s3_client = None
else:
    # Detect the actual bucket region to avoid signature mismatches
    print(f"\n🔍 Detecting S3 bucket region for '{S3_BUCKET_NAME}'...")
    detected_region = get_bucket_region(S3_BUCKET_NAME, aws_access_key, aws_secret_key, aws_region)
    if detected_region != aws_region:
        print(f"  ✓ Bucket region detected: {detected_region} (was configured as: {aws_region})")
        aws_region = detected_region
    else:
        print(f"  ✓ Using configured region: {aws_region}")
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

CHROMADB_S3_KEY = "chromadb_backup/chromadb.tar.gz"  # S3 key for ChromaDB backup
CHROMADB_LOCAL_PATH = "./chroma_db"

def download_chromadb_from_s3():
    """Download ChromaDB backup from S3 if it exists."""
    if s3_client is None:
        print("\n⚠️  S3 client not configured, skipping ChromaDB restore from S3")
        return False
    
    try:
        print("\n📥 Checking for ChromaDB backup in S3...")
        # Check if backup exists
        try:
            s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=CHROMADB_S3_KEY)
            print(f"  ✓ Found ChromaDB backup in S3")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"  ⚠ No ChromaDB backup found in S3 (this is normal for first run)")
                return False
            else:
                raise
        
        # Download the backup
        print(f"  📥 Downloading ChromaDB backup from S3...")
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=CHROMADB_S3_KEY)
        backup_data = response['Body'].read()
        
        # Remove existing chroma_db directory if it exists
        if os.path.exists(CHROMADB_LOCAL_PATH):
            print(f"  🗑️  Removing existing local ChromaDB directory...")
            shutil.rmtree(CHROMADB_LOCAL_PATH)
        
        # Extract the tar.gz file
        print(f"  📦 Extracting ChromaDB backup...")
        with tarfile.open(fileobj=BytesIO(backup_data), mode='r:gz') as tar:
            tar.extractall(path='.')
        
        print(f"  ✓ ChromaDB restored from S3 successfully")
        return True
        
    except Exception as e:
        print(f"  ⚠ Failed to restore ChromaDB from S3: {str(e)}")
        print(f"  → Continuing with empty ChromaDB (this is normal for first run)")
        return False


def upload_chromadb_to_s3(skip_if_extraction_in_progress=False):
    """
    Upload ChromaDB directory to S3 as a backup.
    
    Args:
        skip_if_extraction_in_progress: If True, skip backup if extraction is in progress
    """
    if s3_client is None:
        return False
    
    if not os.path.exists(CHROMADB_LOCAL_PATH):
        return False
    
    # Check if extraction is in progress and skip if requested
    if skip_if_extraction_in_progress:
        with extraction_lock:
            if extraction_in_progress:
                print(f"  ⏭️  Skipping ChromaDB backup (extraction in progress)")
                return False
    
    # Acquire backup lock to prevent concurrent backups
    if not backup_lock.acquire(blocking=False):
        print(f"  ⏭️  Skipping ChromaDB backup (another backup in progress)")
        return False
    
    try:
        print(f"\n📤 Backing up ChromaDB to S3...")
        
        # Create a temporary tar.gz file in memory
        tar_buffer = BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            tar.add(CHROMADB_LOCAL_PATH, arcname=os.path.basename(CHROMADB_LOCAL_PATH))
        
        tar_buffer.seek(0)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=CHROMADB_S3_KEY,
            Body=tar_buffer.getvalue(),
            ContentType='application/gzip'
        )
        
        print(f"  ✓ ChromaDB backed up to S3 successfully")
        return True
        
    except Exception as e:
        print(f"  ⚠ Failed to backup ChromaDB to S3: {str(e)}")
        return False
    finally:
        backup_lock.release()


# Ensure chroma_db directory exists (create if it doesn't)
os.makedirs(CHROMADB_LOCAL_PATH, exist_ok=True)

# Global variables for RAG system
embedding_model = None
chroma_client = None
collection = None
_models_initialized = False
_models_init_lock = threading.Lock()

# Chunk settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Download ChromaDB from S3 in background thread (non-blocking)
def download_chromadb_background():
    """Download ChromaDB from S3 in background thread."""
    try:
        print("📥 Starting ChromaDB download from S3 in background...")
        download_chromadb_from_s3()
        print("✅ ChromaDB download from S3 completed")
    except Exception as e:
        print(f"⚠️ Error downloading ChromaDB from S3: {e}")

# Start ChromaDB download in background thread (non-blocking)
threading.Thread(target=download_chromadb_background, daemon=True).start()

# Initialize ChromaDB at module level (fast, non-blocking)
try:
    chroma_client = chromadb.PersistentClient(
        path=CHROMADB_LOCAL_PATH,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    print(f"✅ ChromaDB client initialized at: {CHROMADB_LOCAL_PATH}")
except Exception as e:
    print(f"⚠️ Error initializing ChromaDB client: {e}")
    chroma_client = None


def ensure_models_initialized():
    """Ensure models are initialized (lazy initialization)."""
    global embedding_model, collection, _models_initialized
    
    if _models_initialized:
        return
    
    with _models_init_lock:
        if _models_initialized:
            return
        
        try:
            print("🚀 Initializing models (lazy initialization)...")
            
            # Initialize embedding model
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded")
            
            # Initialize collection if ChromaDB client is available
            if chroma_client is not None:
                try:
                    existing_collection = chroma_client.get_collection(name="pdf_documents")
                    # Test if collection accepts our embedding dimension
                    test_embedding = embedding_model.encode(["test"])[0].tolist()
                    try:
                        existing_collection.add(
                            documents=["test"],
                            embeddings=[test_embedding],
                            metadatas=[{"test": "true"}],
                            ids=["dimension_test"]
                        )
                        existing_collection.delete(ids=["dimension_test"])
                        collection = existing_collection
                        existing_count = collection.count()
                        print(f"✅ Existing collection loaded with {existing_count} chunks")
                    except Exception as e:
                        error_msg = str(e)
                        if "dimension" in error_msg.lower() or "1536" in error_msg or "384" in error_msg:
                            print(f"🔄 Collection has wrong embedding dimension. Resetting...")
                            try:
                                chroma_client.delete_collection(name="pdf_documents")
                            except:
                                pass
                            collection = chroma_client.create_collection(
                                name="pdf_documents",
                                metadata={"hnsw:space": "cosine"}
                            )
                            print("✅ Collection recreated with SentenceTransformer embeddings (384 dimensions)")
                        else:
                            raise
                except Exception as e:
                    if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                        collection = chroma_client.create_collection(
                            name="pdf_documents",
                            metadata={"hnsw:space": "cosine"}
                        )
                        print("✅ New collection created")
                    else:
                        print(f"⚠️  Error accessing collection: {str(e)}")
                        try:
                            chroma_client.delete_collection(name="pdf_documents")
                        except:
                            pass
                        collection = chroma_client.create_collection(
                            name="pdf_documents",
                            metadata={"hnsw:space": "cosine"}
                        )
                        print("✅ Collection recreated")
            else:
                print("⚠️  ChromaDB client not available, collection not initialized")
            
            _models_initialized = True
            print("✅ Models initialization complete")
        except Exception as e:
            print(f"⚠️ Error initializing models: {e}")
            raise


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def add_documents_to_chromadb(pdf_text_dict):
    """Process documents, create chunks, embeddings, and add to ChromaDB."""
    # Ensure models are initialized (lazy initialization)
    ensure_models_initialized()
    
    global collection, embedding_model
    
    if collection is None or embedding_model is None:
        raise Exception("Collection or embedding model not initialized")
    
    print(f"\n{'='*80}")
    print("🔄 Processing documents for ChromaDB")
    print(f"{'='*80}\n")
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for page_identifier, text in pdf_text_dict.items():
        if not text.strip():
            continue
        
        pdf_name, page_num = page_identifier.split('&')
        
        # Create chunks
        chunks = chunk_text(text)
        print(f"  📄 {page_identifier}: {len(chunks)} chunks created")
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{page_identifier}_chunk_{chunk_idx}"
            
            all_chunks.append(chunk)
            all_metadatas.append({
                "pdf_name": pdf_name,
                "page_num": page_num,
                "page_identifier": page_identifier,
                "chunk_index": chunk_idx,
                "chunk_text": chunk
            })
            all_ids.append(chunk_id)
    
    # Generate embeddings
    print(f"\n  🧮 Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
    embeddings_list = [emb.tolist() for emb in embeddings]
    
    # Add to ChromaDB
    print(f"  💾 Adding documents to ChromaDB...")
    collection.add(
        documents=all_chunks,
        embeddings=embeddings_list,
        metadatas=all_metadatas,
        ids=all_ids
    )
    
    print(f"\n{'='*80}")
    print(f"✅ Successfully added {len(all_chunks)} chunks to ChromaDB")
    print(f"{'='*80}\n")
    
    return len(all_chunks)

# Perform initial backup after initialization
if s3_client is not None:
    print("\n📤 Performing initial ChromaDB backup to S3...")
    threading.Thread(target=upload_chromadb_to_s3, daemon=True).start()


def periodic_chromadb_backup():
    """Periodically backup ChromaDB to S3 every 5 minutes. Skips if extraction is in progress."""
    while True:
        time.sleep(300)  # 5 minutes
        if s3_client is not None:
            # Skip backup if extraction is in progress to avoid conflicts
            upload_chromadb_to_s3(skip_if_extraction_in_progress=True)


# Start periodic backup thread
if s3_client is not None:
    backup_thread = threading.Thread(target=periodic_chromadb_backup, daemon=True)
    backup_thread.start()
    print("✓ Periodic ChromaDB backup thread started (every 5 minutes)")

# Initialize MongoDB client (with error handling)
mongodb_uri = os.getenv("MONGODB_URI", "")
if not mongodb_uri:
    print("\n⚠️  WARNING: MONGODB_URI not found in .env file or environment!")
    print("The brochure tracking functionality will not work without MongoDB URI.")
    mongodb_client = None
elif not MONGODB_AVAILABLE:
    print("\n⚠️  WARNING: pymongo not installed!")
    print("The brochure tracking functionality will not work without pymongo.")
    mongodb_client = None
else:
    try:
        mongodb_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        # Test the connection
        mongodb_client.admin.command('ping')
        print("\n✓ MongoDB connection established")
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"\n⚠️  WARNING: Failed to connect to MongoDB: {e}")
        print("The brochure tracking functionality will not work.")
        mongodb_client = None
    except Exception as e:
        print(f"\n⚠️  WARNING: Error connecting to MongoDB: {e}")
        mongodb_client = None

# Pydantic model for query request
class QueryRequest(BaseModel):
    query: str
    top_k: int = 15


class QueryResponse(BaseModel):
    chunks: List[dict]
    total_results: int

# Pydantic model for calendar request
class CalendarRequest(BaseModel):
    title: str
    date: str  # Format: YYYY-MM-DD
    start_time: str  # Format: HH:MM
    end_time: str  # Format: HH:MM
    description: str = ""
    location: str = ""


def clean_webhook_url(webhook_url: str) -> str:
    """
    Clean and validate webhook URL.
    
    Handles cases where the URL has a redirect parameter and extracts the actual endpoint.
    Example: https://domain.com/?redirect=%2Fapi%2Fwebhook -> https://domain.com/api/webhook
    
    Args:
        webhook_url: The webhook URL to clean
        
    Returns:
        The cleaned webhook URL
    """
    if not webhook_url:
        return webhook_url
    
    try:
        # Parse the URL
        parsed = urlparse(webhook_url)
        print(f"  🔍 Parsing webhook URL - scheme: {parsed.scheme}, netloc: {parsed.netloc}, path: {parsed.path}, query: {parsed.query}")
        
        # Check if there's a redirect parameter
        query_params = parse_qs(parsed.query)
        print(f"  🔍 Query params: {query_params}")
        
        if 'redirect' in query_params:
            # Extract the redirect path
            redirect_path = unquote(query_params['redirect'][0])
            print(f"  🔍 Found redirect parameter: {redirect_path}")
            
            # Reconstruct the URL without the redirect parameter
            cleaned_url = f"{parsed.scheme}://{parsed.netloc}{redirect_path}"
            print(f"  🔧 Cleaned webhook URL: {webhook_url} -> {cleaned_url}")
            return cleaned_url
        
        # If no redirect parameter, return as is
        print(f"  ℹ️ No redirect parameter found, using original URL")
        return webhook_url
    except Exception as e:
        print(f"  ⚠️ Error cleaning webhook URL: {e}")
        import traceback
        traceback.print_exc()
        return webhook_url


def send_webhook_notification(webhook_url: str, job_id: str, job_data: dict, max_retries: int = 3):
    """
    Send webhook notification with retry logic.
    
    Args:
        webhook_url: URL to send the webhook to
        job_id: Job identifier
        job_data: Data to send in the webhook
        max_retries: Maximum number of retry attempts
    """
    # Clean the webhook URL first
    cleaned_url = clean_webhook_url(webhook_url)
    print(f"  🔧 Using cleaned URL for request: {cleaned_url}")
    
    for attempt in range(max_retries):
        try:
            print(f"  📡 Sending webhook notification (attempt {attempt + 1}/{max_retries})...")
            print(f"  🔍 POST request to: {cleaned_url}")
            response = requests.post(
                cleaned_url,
                json=job_data,
                headers={"Content-Type": "application/json"},
                timeout=30,
                allow_redirects=False
            )
            response.raise_for_status()
            print(f"  ✓ Webhook notification sent successfully (status: {response.status_code})")
            return True
        except requests.exceptions.RequestException as e:
            print(f"  ⚠ Webhook notification failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                # Exponential backoff: wait 2^attempt seconds
                wait_time = 2 ** attempt
                print(f"  ⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ Webhook notification failed after {max_retries} attempts")
                return False
    return False


def normalize_phone_number(phone_number: str) -> str:
    """Normalize phone number to DB format: '91' + 10 digits"""
    # Remove all non-digit characters
    digits_only = ''.join(filter(str.isdigit, phone_number))
    
    # If the number starts with "91" and has more than 10 digits, remove the "91" prefix
    if digits_only.startswith("91") and len(digits_only) > 10:
        phone_number_clean = digits_only[2:]  # Remove first 2 digits (91)
    elif len(digits_only) > 10:
        # If it's longer than 10 digits but doesn't start with 91, take last 10 digits
        phone_number_clean = digits_only[-10:]
    elif len(digits_only) < 10:
        # If less than 10 digits, pad with zeros
        phone_number_clean = digits_only.zfill(10)
    else:
        # Exactly 10 digits
        phone_number_clean = digits_only
    
    # Convert to DB format: "91" + 10 digits
    return f"91{phone_number_clean}" if len(phone_number_clean) == 10 else phone_number


def determine_course_from_query(query: str) -> str:
    """Determine if query is course-specific or general using OpenAI"""
    if not openai_client:
        print("  ⚠ OpenAI client not available, defaulting to 'general'")
        return "general"
    
    try:
        prompt = f"""Analyze the following query and determine if it's related to a SPECIFIC engineering course or is a GENERAL query about the college.

Available courses:
- Computer Science and Engineering
- Information Technology
- Electrical and Electronics Engineering
- Electronics and Communication Engineering
- Mechanical Engineering

If the query is about a SPECIFIC course, return ONLY the exact course name from the list above.
If the query is GENERAL (about college, admissions, fees, facilities, etc. without mentioning a specific course), return "general".

Query: "{query}"

Return ONLY the course name or "general" (no additional text):"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a course classification assistant. Analyze queries and determine if they're course-specific or general."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        course = response.choices[0].message.content.strip()
        
        # Validate course name
        valid_courses = [
            "Computer Science and Engineering",
            "Information Technology",
            "Electrical and Electronics Engineering",
            "Electronics and Communication Engineering",
            "Mechanical Engineering",
            "general"
        ]
        
        # Check if response matches any valid course (case-insensitive)
        course_lower = course.lower()
        for valid_course in valid_courses:
            if valid_course.lower() in course_lower or course_lower in valid_course.lower():
                return valid_course
        
        # If no match, check for partial matches
        if "computer" in course_lower and "science" in course_lower:
            return "Computer Science and Engineering"
        elif "information" in course_lower and "technology" in course_lower:
            return "Information Technology"
        elif "electrical" in course_lower and "electronics" in course_lower:
            return "Electrical and Electronics Engineering"
        elif "electronics" in course_lower and "communication" in course_lower:
            return "Electronics and Communication Engineering"
        elif "mechanical" in course_lower and "engineering" in course_lower:
            return "Mechanical Engineering"
        else:
            return "general"
            
    except Exception as e:
        print(f"  ⚠ Error determining course: {e}, defaulting to 'general'")
        return "general"


def track_brochure_shared(phone_number: str, email: str, compiled_pdf_url: str, query: str):
    """Track brochure sharing in MongoDB brochuresShared collection"""
    if not mongodb_client:
        print("  ⚠ MongoDB client not available, skipping brochure tracking")
        return
    
    if not compiled_pdf_url:
        print("  ⚠ No PDF URL provided, skipping brochure tracking")
        return
    
    # At least one contact method must be provided
    if not phone_number and not email:
        print("  ⚠ No contact information provided, skipping brochure tracking")
        return
    
    try:
        # Normalize phone number to DB format (if provided)
        phone_db_format = None
        if phone_number:
            phone_db_format = normalize_phone_number(phone_number)
        
        # Determine course from query
        print(f"  🔍 Determining course from query...")
        course = determine_course_from_query(query)
        print(f"  ✓ Course determined: {course}")
        
        # Get database and collection
        db = mongodb_client["ALLIANCE"]
        brochures_collection = db["brochuresShared"]
        
        # Create document (only include fields that are provided)
        brochure_doc = {
            "compiled_pdf_url": compiled_pdf_url,
            "course": course
        }
        
        if phone_db_format:
            brochure_doc["phone_number"] = phone_db_format
        if email:
            brochure_doc["email"] = email
        
        # Insert into MongoDB
        result = brochures_collection.insert_one(brochure_doc)
        print(f"  ✓ Brochure sharing tracked in MongoDB (ID: {result.inserted_id})")
        
    except Exception as e:
        print(f"  ⚠ Error tracking brochure sharing: {e}")
        # Don't raise exception - this is a tracking feature, shouldn't break the main flow


def get_calendar_service():
    """Authenticate and return Google Calendar service"""
    if not GOOGLE_CALENDAR_AVAILABLE:
        raise ValueError("Google Calendar libraries not available")
    
    creds = None
    
    # First, try to load token from environment variable (base64 encoded)
    token_base64 = os.getenv('GOOGLE_CALENDAR_TOKEN')
    if token_base64:
        try:
            import base64
            token_data = base64.b64decode(token_base64)
            creds = pickle.loads(token_data)
            print("  ✓ Loaded calendar token from environment variable")
        except Exception as e:
            print(f"  ⚠ Failed to load token from environment: {e}")
            creds = None
    
    # If no token from env, try to load from pickle file (fallback)
    if not creds:
        token_path = os.path.join(os.path.dirname(__file__), 'token.pickle')
        if os.path.exists(token_path):
            try:
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)
                print("  ✓ Loaded calendar token from pickle file")
            except Exception as e:
                print(f"  ⚠ Failed to load token from pickle file: {e}")
                creds = None
    
    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("  🔄 Refreshing expired token...")
            creds.refresh(Request())
            print("  ✓ Token refreshed successfully")
        else:
            # Load credentials from environment variable
            credentials_json = os.getenv('GOOGLE_CALENDAR_CREDENTIALS')
            
            if not credentials_json:
                raise ValueError("GOOGLE_CALENDAR_CREDENTIALS not found in .env file")
            
            # Parse JSON and create flow
            import json
            credentials_dict = json.loads(credentials_json)
            flow = InstalledAppFlow.from_client_config(
                credentials_dict, CALENDAR_SCOPES)
            print("  🔐 Starting OAuth flow...")
            creds = flow.run_local_server(port=0)
            print("  ✓ OAuth authentication completed")
        
        # Save credentials for future use (both to env format and pickle file)
        # Note: The base64 token should be updated in .env manually after first auth
        token_path = os.path.join(os.path.dirname(__file__), 'token.pickle')
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
        print("  ✓ Token saved to pickle file")
    
    return build('calendar', 'v3', credentials=creds)


def check_calendar_conflicts(date: str, start_time: str, end_time: str, timezone: str = 'Asia/Kolkata'):
    """
    Check if there are any events during the specified time period
    
    Args:
        date: Date in 'YYYY-MM-DD' format
        start_time: Time in 'HH:MM' format
        end_time: Time in 'HH:MM' format
        timezone: Timezone (default: 'Asia/Kolkata')
    
    Returns:
        List of conflicting events with details
    """
    try:
        service = get_calendar_service()
        
        # Build datetime strings in RFC3339 format
        # For Asia/Kolkata timezone, we need to add +05:30 offset
        time_min = f"{date}T{start_time}:00+05:30"
        time_max = f"{date}T{end_time}:00+05:30"
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        if not events:
            return []
        
        # Format conflicting events for response
        conflicts = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            conflict_info = {
                'title': event.get('summary', 'Untitled Event'),
                'start': start,
                'end': end,
                'location': event.get('location', ''),
                'description': event.get('description', '')
            }
            conflicts.append(conflict_info)
        
        return conflicts
        
    except Exception as e:
        print(f"✗ Error checking conflicts: {e}")
        raise


def create_calendar_event(title: str, date: str, start_time: str, end_time: str, 
                         description: str = '', location: str = '', timezone: str = 'Asia/Kolkata'):
    """
    Create a calendar event
    
    Args:
        title: Event name
        date: Date in 'YYYY-MM-DD' format (e.g., '2024-12-01')
        start_time: Time in 'HH:MM' format (e.g., '14:00')
        end_time: Time in 'HH:MM' format (e.g., '15:00')
        description: Event description
        location: Event location
        timezone: Timezone (default: 'Asia/Kolkata')
    
    Returns:
        Created event object
    """
    try:
        service = get_calendar_service()
        
        # Build datetime strings
        start_datetime = f"{date}T{start_time}:00"
        end_datetime = f"{date}T{end_time}:00"
        
        event = {
            'summary': title,
            'location': location,
            'description': description,
            'start': {
                'dateTime': start_datetime,
                'timeZone': timezone,
            },
            'end': {
                'dateTime': end_datetime,
                'timeZone': timezone,
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'popup', 'minutes': 30},
                    {'method': 'email', 'minutes': 1440},  # 24 hours before
                ],
            },
        }
        
        created_event = service.events().insert(calendarId='primary', body=event).execute()
        return created_event
        
    except Exception as e:
        print(f"✗ Error creating event: {e}")
        raise


def extract_text_with_ocr(page, page_num, pdf_name):
    """Extract text from a PDF page using OCR."""
    global TESSERACT_FOUND
    
    if not TESSERACT_FOUND:
        print(f"    ⚠ OCR skipped (Tesseract not available)")
        return ""
    
    try:
        print(f"    🔍 Running OCR on {pdf_name}&{page_num}...", end="", flush=True)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))
        ocr_text = pytesseract.image_to_string(img, lang='eng')
        print(" ✓ Done")
        return ocr_text.strip()
    except pytesseract.TesseractNotFoundError:
        TESSERACT_FOUND = False
        print(" ✗ Failed (Tesseract not found)")
        return ""
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return ""


def extract_text_from_pdf(pdf_bytes, pdf_filename, use_ocr=True):
    """Extract text from PDF file page by page."""
    pdf_text = {}
    page_objects = {}  # Store page objects for S3 upload
    
    try:
        pdf_name = Path(pdf_filename).stem
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        print(f"\n{'='*80}")
        print(f"📄 Processing PDF: {pdf_filename}")
        print(f"📊 Total pages: {len(doc)}")
        print(f"🔧 OCR enabled: {use_ocr}")
        print(f"{'='*80}\n")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_identifier = f"{pdf_name}&{page_num + 1}"
            
            print(f"  📖 Page {page_num + 1}/{len(doc)} ({page_identifier})")
            
            print(f"    📝 Extracting regular text...", end="", flush=True)
            regular_text = page.get_text()
            regular_char_count = len(regular_text.strip())
            print(f" ✓ ({regular_char_count} chars)")
            
            ocr_text = ""
            ocr_char_count = 0
            if use_ocr:
                ocr_text = extract_text_with_ocr(page, page_num + 1, pdf_name)
                ocr_char_count = len(ocr_text.strip())
                if ocr_text:
                    print(f"      ✓ OCR extracted {ocr_char_count} chars")
            
            if regular_text.strip() and ocr_text.strip():
                if regular_text.strip() in ocr_text or len(regular_text.strip()) < 50:
                    combined_text = ocr_text
                    print(f"    💡 Using OCR text only (regular text contained in OCR)")
                else:
                    combined_text = f"{regular_text}\n\n{ocr_text}"
                    print(f"    💡 Combined regular + OCR text")
            elif ocr_text.strip():
                combined_text = ocr_text
                print(f"    💡 Using OCR text only")
            else:
                combined_text = regular_text
                print(f"    💡 Using regular text only")
            
            pdf_text[page_identifier] = combined_text
            page_objects[page_identifier] = page
            
            total_chars = len(combined_text.strip())
            print(f"    ✅ Page {page_identifier} complete - Total: {total_chars} chars")
            print(f"    {'-'*76}")
        
        print(f"\n{'='*80}")
        print(f"✅ PDF Processing Complete: {pdf_filename}")
        print(f"📊 Total pages processed: {len(pdf_text)}")
        print(f"{'='*80}\n")
        
        return pdf_text, page_objects, doc
        
    except Exception as e:
        print(f"\n❌ Error extracting text from {pdf_filename}: {str(e)}\n")
        raise Exception(f"Error extracting text from {pdf_filename}: {str(e)}")


def store_in_chromadb(page_identifier, text, pdf_name, page_number):
    """Store extracted text in ChromaDB with metadata. Prevents duplicates."""
    # Ensure models are initialized (lazy initialization)
    ensure_models_initialized()
    
    global collection
    if collection is None:
        print(f" ✗ Failed (Collection not initialized)")
        return False
    
    try:
        print(f"    💾 Storing {page_identifier} in ChromaDB...", end="", flush=True)
        
        # Check if this page already exists (prevent duplicates)
        existing = collection.get(
            where={"page_identifier": page_identifier},
            limit=1
        )
        
        is_new_page = not existing['ids']
        
        if existing['ids']:
            # Page exists, update it instead of creating duplicate
            print(f" (updating existing)...", end="", flush=True)
            collection.update(
                ids=[existing['ids'][0]],
                documents=[text],
                metadatas=[{
                    "page_identifier": page_identifier,
                    "pdf_name": pdf_name,
                    "page_number": str(page_number),
                    "char_count": str(len(text))
                }]
            )
            print(" ✓ Updated")
        else:
            # New page, add it
            doc_id = f"{page_identifier}"
            collection.add(
                documents=[text],
                metadatas=[{
                    "page_identifier": page_identifier,
                    "pdf_name": pdf_name,
                    "page_number": str(page_number),
                    "char_count": str(len(text))
                }],
                ids=[doc_id]
            )
            print(" ✓ Stored")
            
            # Update PDF names cache when new PDF is added
            global _cached_pdf_names, _pdf_names_cache_lock
            with _pdf_names_cache_lock:
                _cached_pdf_names.add(pdf_name)
        
        # Note: Backup is triggered after batch extraction completes, not after each page
        # This prevents race conditions and reduces S3 API calls
        
        return True
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return False


def upload_page_to_s3(page, page_identifier):
    """
    Convert PDF page to image and upload to S3.
    
    Args:
        page: PyMuPDF page object
        page_identifier: Page identifier (e.g., "E-Brochure-1&3")
    
    Returns:
        str: S3 URL of uploaded image, or None if failed
    """
    if s3_client is None:
        print(f"    ⚠ S3 client not configured, skipping upload for {page_identifier}")
        return None
    
    try:
        print(f"    📤 Uploading {page_identifier} to S3...", end="", flush=True)
        
        # Render page as high-quality image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        
        # S3 key (filename in bucket)
        s3_key = f"{page_identifier}.png"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=img_bytes,
            ContentType='image/png'
        )
        
        # Generate S3 URL
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
        
        print(f" ✓ Uploaded")
        return s3_url
        
    except ClientError as e:
        print(f" ✗ Failed (S3 Error: {e.response['Error']['Message']})")
        return None
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return None


def upload_full_pdf_to_s3(pdf_bytes: bytes, pdf_filename: str):
    """
    Upload the entire PDF file to S3.
    
    Args:
        pdf_bytes: PDF file contents as bytes
        pdf_filename: Original filename of the PDF
    
    Returns:
        str: S3 URL of uploaded PDF, or None if failed
    """
    if s3_client is None:
        print(f"    ⚠ S3 client not configured, skipping full PDF upload for {pdf_filename}")
        return None
    
    try:
        print(f"    📤 Uploading full PDF {pdf_filename} to S3...", end="", flush=True)
        
        # Sanitize filename for S3 key (remove special characters, keep extension)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(c for c in pdf_filename if c.isalnum() or c in ('-', '_', '.'))
        if not safe_filename.endswith('.pdf'):
            safe_filename = f"{safe_filename}.pdf"
        
        # S3 key (store in pdfs/ subfolder for organization)
        s3_key = f"pdfs/{timestamp}_{safe_filename}"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=pdf_bytes,
            ContentType='application/pdf',
            Metadata={
                'original_filename': pdf_filename,
                'uploaded_at': timestamp
            }
        )
        
        # Generate S3 URL
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
        
        print(f" ✓ Uploaded")
        return s3_url
        
    except ClientError as e:
        print(f" ✗ Failed (S3 Error: {e.response['Error']['Message']})")
        return None
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return None


def create_compiled_pdf_from_images(s3_image_urls, user_number, query):
    """
    Download images from S3, compile them into a single PDF, and upload to S3.
    
    Args:
        s3_image_urls: List of S3 image URLs
        user_number: User's phone number
        query: The original query
        
    Returns:
        str: S3 URL of compiled PDF, or None if failed
    """
    if s3_client is None:
        print(f"  ⚠ S3 client not configured, cannot create compiled PDF")
        return None
    
    try:
        print(f"  📄 Creating compiled PDF from {len(s3_image_urls)} images...")
        
        if not s3_image_urls:
            return None
        
        # Create a temporary PDF file
        import tempfile
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_pdf_path = temp_pdf.name
        temp_pdf.close()
        
        # Create PDF with ReportLab
        c = canvas.Canvas(temp_pdf_path, pagesize=A4)
        page_width, page_height = A4
        
        # Download and add each image to PDF
        for idx, img_url in enumerate(s3_image_urls, 1):
            print(f"    📥 Processing image {idx}/{len(s3_image_urls)}...", end="", flush=True)
            
            try:
                # Download image from S3
                response = requests.get(img_url, timeout=30)
                response.raise_for_status()
                
                # Open image with PIL
                img = Image.open(BytesIO(response.content))
                
                # Calculate dimensions to fit page while maintaining aspect ratio
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                
                # Fit to page with margins
                margin = 20
                available_width = page_width - (2 * margin)
                available_height = page_height - (2 * margin)
                
                if available_width * aspect <= available_height:
                    # Width is limiting factor
                    display_width = available_width
                    display_height = available_width * aspect
                else:
                    # Height is limiting factor
                    display_height = available_height
                    display_width = available_height / aspect
                
                # Center image on page
                x = (page_width - display_width) / 2
                y = (page_height - display_height) / 2
                
                # Draw image
                img_reader = ImageReader(BytesIO(response.content))
                c.drawImage(img_reader, x, y, width=display_width, height=display_height)
                
                # Add new page if not last image
                if idx < len(s3_image_urls):
                    c.showPage()
                
                print(" ✓")
                
            except Exception as e:
                print(f" ✗ Failed: {str(e)}")
                continue
        
        # Save PDF
        c.save()
        print(f"  ✓ PDF created successfully")
        
        # Upload to S3
        print(f"  📤 Uploading compiled PDF to S3...", end="", flush=True)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"query_{user_number}_{timestamp}.pdf"
        
        # Read PDF file
        with open(temp_pdf_path, 'rb') as pdf_file:
            pdf_bytes = pdf_file.read()
        
        # Upload to S3 (using the same bucket as page images)
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"compiled_pdfs/{pdf_filename}",  # Store in a subfolder for organization
            Body=pdf_bytes,
            ContentType='application/pdf',
            Metadata={
                'user_number': user_number,
                'query': query[:200],  # Truncate long queries
                'page_count': str(len(s3_image_urls))
            }
        )
        
        # Generate direct S3 object URL (region-specific format)
        # Format: https://{bucket}.s3.{region}.amazonaws.com/{key}
        compiled_pdf_url = f"https://{S3_BUCKET_NAME}.s3.{aws_region}.amazonaws.com/compiled_pdfs/{pdf_filename}"
        print(f" ✓ Uploaded")
        print(f"  ✓ Compiled PDF URL: {compiled_pdf_url}")
        
        # Clean up temp file
        os.unlink(temp_pdf_path)
        
        return compiled_pdf_url
        
    except Exception as e:
        print(f"  ✗ Failed to create compiled PDF: {str(e)}")
        return None


def process_extraction_job(job_id: str, files_data: list, use_ocr: bool, webhook_url: Optional[str] = None):
    """
    Background function to process PDF extraction.
    
    Args:
        job_id: Unique job identifier
        files_data: List of tuples (filename, file_contents_bytes)
        use_ocr: Whether to use OCR
        webhook_url: Optional webhook URL to notify when complete
    """
    # Set extraction flag to prevent concurrent backups
    global extraction_in_progress
    with extraction_lock:
        extraction_in_progress = True
    
    try:
        with job_lock:
            extraction_jobs[job_id]["status"] = JobStatus.PROCESSING
            extraction_jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        print("\n" + "="*80)
        print(f"🚀 Starting batch PDF extraction (Job ID: {job_id})")
        print(f"📁 Total files: {len(files_data)}")
        print(f"🔧 OCR enabled: {use_ocr}")
        print(f"🔧 OCR available: {TESSERACT_FOUND}")
        print("="*80)
        
        results = {}
        errors = []
        stored_count = 0
        s3_upload_count = 0
        pdf_s3_urls = {}  # Store full PDF S3 URLs by filename
        
        for idx, (filename, contents) in enumerate(files_data, 1):
            print(f"\n📦 Processing file {idx}/{len(files_data)}: {filename}")
            
            if not filename.lower().endswith('.pdf'):
                error_msg = f"{filename}: Not a PDF file"
                errors.append(error_msg)
                print(f"  ❌ {error_msg}")
                continue
            
            try:
                print(f"  📥 File size: {len(contents)} bytes")
                
                # Upload full PDF to S3 first
                full_pdf_s3_url = upload_full_pdf_to_s3(contents, filename)
                if full_pdf_s3_url:
                    pdf_s3_urls[filename] = full_pdf_s3_url
                
                # Extract text
                extracted_text, page_objects, doc = extract_text_from_pdf(contents, filename, use_ocr)
                
                # Store each page in ChromaDB and upload to S3
                pdf_name = Path(filename).stem
                print(f"\n  💾 Storing pages in ChromaDB and uploading to S3...")
                
                for page_identifier, text in extracted_text.items():
                    # Extract page number from identifier
                    page_number = int(page_identifier.split('&')[1])
                    
                    # Store in ChromaDB
                    if store_in_chromadb(page_identifier, text, pdf_name, page_number):
                        stored_count += 1
                    
                    # Upload page image to S3
                    page = page_objects[page_identifier]
                    s3_url = upload_page_to_s3(page, page_identifier)
                    if s3_url:
                        s3_upload_count += 1
                
                # Close the document
                doc.close()
                
                results.update(extracted_text)
                print(f"  ✅ Successfully extracted and stored {len(extracted_text)} pages from {filename}")
                
                # Send progress webhook notification after each file is processed
                if webhook_url:
                    progress_data = {
                        "job_id": job_id,
                        "status": "processing",
                        "current_file": filename,
                        "current_file_index": idx,
                        "total_files": len(files_data),
                        "files_processed": idx,
                        "total_pages_extracted": len(results),
                        "total_pages_stored_in_db": stored_count,
                        "total_pages_uploaded_to_s3": s3_upload_count,
                        "progress_percentage": int((idx / len(files_data)) * 100),
                        "ocr_enabled": use_ocr,
                        "ocr_available": TESSERACT_FOUND,
                        "errors": errors.copy() if errors else []
                    }
                    print(f"  📡 Sending progress webhook notification...")
                    send_webhook_notification(webhook_url, job_id, progress_data)
                
            except Exception as e:
                error_msg = f"{filename}: {str(e)}"
                errors.append(error_msg)
                print(f"  ❌ Failed: {error_msg}")
                
                # Send progress webhook notification even on error (to report the error)
                if webhook_url:
                    progress_data = {
                        "job_id": job_id,
                        "status": "processing",
                        "current_file": filename,
                        "current_file_index": idx,
                        "total_files": len(files_data),
                        "files_processed": idx,
                        "total_pages_extracted": len(results),
                        "total_pages_stored_in_db": stored_count,
                        "total_pages_uploaded_to_s3": s3_upload_count,
                        "progress_percentage": int((idx / len(files_data)) * 100),
                        "ocr_enabled": use_ocr,
                        "ocr_available": TESSERACT_FOUND,
                        "errors": errors.copy() if errors else [],
                        "last_error": error_msg
                    }
                    print(f"  📡 Sending progress webhook notification (with error)...")
                    send_webhook_notification(webhook_url, job_id, progress_data)
        
        # Backup ChromaDB to S3 after extraction completes
        if stored_count > 0 and s3_client is not None:
            print("📤 Backing up ChromaDB after extraction completes...")
            def delayed_backup():
                time.sleep(2)  # Wait 2 seconds for ChromaDB to flush writes
                upload_chromadb_to_s3()
            threading.Thread(target=delayed_backup, daemon=True).start()
        
        # Prepare response data
        response_data = {
            "job_id": job_id,
            "status": "success" if results else "failed",
            "total_files_processed": len(files_data),
            "total_pages_extracted": len(results),
            "total_pages_stored_in_db": stored_count,
            "total_pages_uploaded_to_s3": s3_upload_count,
            "ocr_enabled": use_ocr,
            "ocr_available": TESSERACT_FOUND,
            "page_identifiers": list(results.keys()),
            "pdf_s3_urls": pdf_s3_urls  # Full PDF S3 URLs
        }
        
        if errors:
            response_data["errors"] = errors
        
        # Update job status
        with job_lock:
            extraction_jobs[job_id]["status"] = JobStatus.COMPLETED if results else JobStatus.FAILED
            extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            extraction_jobs[job_id]["result"] = response_data
        
        print("\n" + "="*80)
        print(f"🎉 Batch extraction complete! (Job ID: {job_id})")
        print(f"✅ Success: {len(results)} pages extracted")
        print(f"💾 Stored: {stored_count} pages in ChromaDB")
        print(f"📤 Uploaded: {s3_upload_count} pages to S3")
        if errors:
            print(f"⚠ Errors: {len(errors)} files failed")
        print("="*80 + "\n")
        
        # Send webhook notification if provided
        if webhook_url:
            print(f"  🔔 Sending webhook notification to {webhook_url}...")
            send_webhook_notification(webhook_url, job_id, response_data)
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Job {job_id} failed: {error_msg}\n")
        
        # Update job status to failed
        with job_lock:
            extraction_jobs[job_id]["status"] = JobStatus.FAILED
            extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            extraction_jobs[job_id]["error"] = error_msg
            extraction_jobs[job_id]["result"] = {
                "job_id": job_id,
                "status": "failed",
                "error": error_msg
            }
        
        # Send webhook notification for failure
        if webhook_url:
            print(f"  🔔 Sending failure webhook notification to {webhook_url}...")
            failure_data = {
                "job_id": job_id,
                "status": "failed",
                "error": error_msg
            }
            send_webhook_notification(webhook_url, job_id, failure_data)
    finally:
        # Always reset extraction flag when done
        with extraction_lock:
            extraction_in_progress = False


@app.post("/extract")
async def extract_text(
    files: List[UploadFile] = File(...),
    use_ocr: bool = Form(True),
    webhook_url: Optional[str] = Form(None)
):
    """
    Extract text from multiple PDF files and store in ChromaDB.
    
    If webhook_url is provided, the extraction will be processed asynchronously
    and a webhook will be called when complete. Otherwise, it processes synchronously.
    
    Args:
        files: List of PDF files to extract
        use_ocr: Whether to use OCR for text extraction
        webhook_url: Optional webhook URL to notify when extraction is complete
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # If webhook_url is provided, process asynchronously
    if webhook_url:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Read all file contents into memory (required for async processing)
        files_data = []
        for file in files:
            contents = await file.read()
            files_data.append((file.filename, contents))
        
        # Initialize job tracking
        with job_lock:
            extraction_jobs[job_id] = {
                "job_id": job_id,
                "status": JobStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "total_files": len(files),
                "use_ocr": use_ocr,
                "webhook_url": webhook_url,
                "result": None,
                "error": None
            }
        
        # Start background processing
        print(f"\n🔄 Starting async extraction job: {job_id}")
        print(f"📡 Webhook URL: {webhook_url}")
        threading.Thread(
            target=process_extraction_job,
            args=(job_id, files_data, use_ocr, webhook_url),
            daemon=True
        ).start()
        
        # Return immediately with job ID
        return JSONResponse(content={
            "status": "processing",
            "job_id": job_id,
            "message": "Extraction started. You will be notified via webhook when complete.",
            "webhook_url": webhook_url,
            "check_status_url": f"/extract/status/{job_id}"
        })
    
    # Synchronous processing (original behavior when no webhook)
    # Set extraction flag to prevent concurrent backups
    global extraction_in_progress
    with extraction_lock:
        extraction_in_progress = True
    
    try:
        print("\n" + "="*80)
        print(f"🚀 Starting batch PDF extraction (synchronous)")
        print(f"📁 Total files received: {len(files)}")
        print(f"🔧 OCR enabled: {use_ocr}")
        print(f"🔧 OCR available: {TESSERACT_FOUND}")
        print("="*80)
        
        total_chunks = 0
        processed_files = []
        
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                continue
            
            # Read PDF bytes
            pdf_bytes = await file.read()
            
            # Extract text from PDF
            pdf_text_dict, page_objects, doc = extract_text_from_pdf(
                pdf_bytes, 
                file.filename, 
                use_ocr=use_ocr
            )
            
            # Add to ChromaDB
            chunks_added = add_documents_to_chromadb(pdf_text_dict)
            total_chunks += chunks_added
            
            processed_files.append({
                "filename": file.filename,
                "pages": len(pdf_text_dict),
                "chunks": chunks_added
            })
            
            doc.close()
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Processed {len(processed_files)} PDF(s)",
            "total_chunks": total_chunks,
            "files": processed_files
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always reset extraction flag when done
        with extraction_lock:
            extraction_in_progress = False


@app.get("/extract/status/{job_id}")
async def get_extraction_status(job_id: str):
    """
    Get the status of an extraction job.
    
    Args:
        job_id: The job identifier returned from the /extract endpoint
    
    Returns:
        Job status and result (if completed)
    """
    with job_lock:
        if job_id not in extraction_jobs:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        job = extraction_jobs[job_id]
        
        response = {
            "job_id": job_id,
            "status": job["status"].value,
            "created_at": job["created_at"],
            "total_files": job["total_files"],
            "use_ocr": job["use_ocr"],
            "webhook_url": job.get("webhook_url")
        }
        
        if "started_at" in job:
            response["started_at"] = job["started_at"]
        
        if "completed_at" in job:
            response["completed_at"] = job["completed_at"]
        
        if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
            response["result"] = job["result"]
            if job.get("error"):
                response["error"] = job["error"]
        
        return JSONResponse(content=response)


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question.
    Returns top K most similar chunks using cosine similarity.
    """
    try:
        # Ensure models are initialized (lazy initialization)
        ensure_models_initialized()
        
        global collection, embedding_model
        
        if collection is None or embedding_model is None:
            raise HTTPException(status_code=500, detail="Collection or embedding model not initialized")
        
        # Generate query embedding
        query_embedding = embedding_model.encode([request.query])[0].tolist()
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.top_k, 15)  # Max 15 as specified
        )
        
        # Format results
        chunks = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                chunks.append({
                    "chunk_text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None,
                    "id": results['ids'][0][i]
                })
        
        return QueryResponse(
            chunks=chunks,
            total_results=len(chunks)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calendar")
async def create_calendar_event_endpoint(request: CalendarRequest):
    """
    Create a calendar event with conflict checking.
    If there are overlapping events, returns an error with conflict details.
    Only creates the event if there are no conflicts.
    """
    if not GOOGLE_CALENDAR_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Google Calendar libraries not available. Please install required packages."
        )
    
    print("\n" + "="*80)
    print(f"📅 Creating calendar event")
    print(f"  Title: {request.title}")
    print(f"  Date: {request.date}")
    print(f"  Time: {request.start_time} - {request.end_time}")
    print(f"  Location: {request.location}")
    print("="*80)
    
    try:
        # First, check for conflicts
        print(f"  🔍 Checking for conflicts...")
        conflicts = check_calendar_conflicts(
            date=request.date,
            start_time=request.start_time,
            end_time=request.end_time
        )
        
        if conflicts:
            print(f"  ⚠ Found {len(conflicts)} conflicting event(s)")
            conflict_details = []
            for conflict in conflicts:
                conflict_details.append({
                    "title": conflict['title'],
                    "start": conflict['start'],
                    "end": conflict['end'],
                    "location": conflict.get('location', '')
                })
                print(f"    - {conflict['title']} ({conflict['start']} to {conflict['end']})")
            
            return JSONResponse(
                status_code=409,  # Conflict status code
                content={
                    "status": "conflict",
                    "message": f"There are {len(conflicts)} overlapping event(s) at this time. Please choose a different time slot.",
                    "conflicts": conflict_details,
                    "requested_time": {
                        "date": request.date,
                        "start_time": request.start_time,
                        "end_time": request.end_time
                    }
                }
            )
        
        # No conflicts, proceed to create the event
        print(f"  ✓ No conflicts found, creating event...")
        created_event = create_calendar_event(
            title=request.title,
            date=request.date,
            start_time=request.start_time,
            end_time=request.end_time,
            description=request.description,
            location=request.location
        )
        
        print(f"  ✓ Event created successfully!")
        print(f"    → Link: {created_event.get('htmlLink')}")
        print(f"    → Event ID: {created_event.get('id')}")
        
        return JSONResponse(content={
            "status": "success",
            "message": "Event created successfully",
            "event": {
                "id": created_event.get('id'),
                "title": created_event.get('summary'),
                "start": created_event['start'].get('dateTime', created_event['start'].get('date')),
                "end": created_event['end'].get('dateTime', created_event['end'].get('date')),
                "location": created_event.get('location', ''),
                "htmlLink": created_event.get('htmlLink')
            }
        })
        
    except ValueError as e:
        print(f"  ❌ Error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create calendar event: {str(e)}"
        )


@app.get("/pdfs")
async def list_pdfs():
    """
    List all PDFs stored in ChromaDB with their metadata.
    Returns information about each PDF including name, page count, and total characters.
    """
    try:
        print("\n" + "="*80)
        print(f"📋 Listing all PDFs in ChromaDB...")
        print("="*80)
        
        # Get all documents from ChromaDB
        # Use a large limit to get all documents, or paginate if needed
        all_results = collection.get(limit=10000)  # Adjust limit as needed
        
        if not all_results.get('metadatas') or len(all_results['metadatas']) == 0:
            print("  ⚠ No PDFs found in ChromaDB")
            return JSONResponse(content={
                "status": "success",
                "total_pdfs": 0,
                "pdfs": []
            })
        
        # Group by pdf_name
        pdf_info = {}
        
        for metadata in all_results['metadatas']:
            pdf_name = metadata.get('pdf_name', 'unknown')
            page_number = int(metadata.get('page_number', 0))
            char_count = int(metadata.get('char_count', 0))
            
            if pdf_name not in pdf_info:
                pdf_info[pdf_name] = {
                    "pdf_name": pdf_name,
                    "page_count": 0,
                    "total_characters": 0,
                    "pages": []
                }
            
            pdf_info[pdf_name]["page_count"] += 1
            pdf_info[pdf_name]["total_characters"] += char_count
            pdf_info[pdf_name]["pages"].append({
                "page_number": page_number,
                "page_identifier": metadata.get('page_identifier', ''),
                "characters": char_count
            })
        
        # Sort pages by page number for each PDF
        for pdf_name in pdf_info:
            pdf_info[pdf_name]["pages"].sort(key=lambda x: x["page_number"])
        
        # Convert to list and sort by PDF name
        pdfs_list = list(pdf_info.values())
        pdfs_list.sort(key=lambda x: x["pdf_name"])
        
        print(f"  ✓ Found {len(pdfs_list)} PDF(s) in ChromaDB")
        for pdf in pdfs_list:
            print(f"    📄 {pdf['pdf_name']}: {pdf['page_count']} pages, {pdf['total_characters']:,} characters")
        
        print("="*80 + "\n")
        
        return JSONResponse(content={
            "status": "success",
            "total_pdfs": len(pdfs_list),
            "pdfs": pdfs_list
        })
        
    except Exception as e:
        print(f"  ❌ Error listing PDFs: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Failed to list PDFs: {str(e)}")


@app.delete("/pdfs/{pdf_name}")
async def delete_pdf(pdf_name: str, delete_s3_images: bool = False):
    """
    Delete a PDF and all its pages from ChromaDB.
    
    Args:
        pdf_name: The name of the PDF to delete (as stored in ChromaDB metadata)
        delete_s3_images: If True, also delete corresponding S3 images (default: False)
    
    Returns:
        Information about what was deleted
    """
    try:
        print("\n" + "="*80)
        print(f"🗑️  Deleting PDF: {pdf_name}")
        print(f"   Delete S3 images: {delete_s3_images}")
        print("="*80)
        
        # First, get all pages for this PDF to see what we're deleting
        pdf_results = collection.get(
            where={"pdf_name": pdf_name},
            limit=10000
        )
        
        if not pdf_results.get('ids') or len(pdf_results['ids']) == 0:
            print(f"  ⚠ PDF '{pdf_name}' not found in ChromaDB")
            raise HTTPException(
                status_code=404,
                detail=f"PDF '{pdf_name}' not found in ChromaDB"
            )
        
        page_count = len(pdf_results['ids'])
        page_identifiers = []
        
        if pdf_results.get('metadatas'):
            for metadata in pdf_results['metadatas']:
                page_id = metadata.get('page_identifier', '')
                if page_id:
                    page_identifiers.append(page_id)
        
        print(f"  📊 Found {page_count} page(s) to delete")
        
        # Delete from ChromaDB using where clause (deletes all matching documents)
        print(f"  🗑️  Deleting from ChromaDB...", end="", flush=True)
        try:
            collection.delete(
                where={"pdf_name": pdf_name}
            )
            print(" ✓ Deleted")
        except Exception as delete_error:
            # Fallback: delete by IDs if where clause doesn't work
            print(f" ⚠ Delete by where failed, trying by IDs...", end="", flush=True)
            collection.delete(ids=pdf_results['ids'])
            print(" ✓ Deleted")
        
        # Optionally delete S3 images
        s3_deleted_count = 0
        s3_deleted_keys = []
        if delete_s3_images and s3_client is not None:
            print(f"  🗑️  Deleting S3 images...")
            for page_identifier in page_identifiers:
                s3_key = f"{page_identifier}.png"
                try:
                    s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                    s3_deleted_count += 1
                    s3_deleted_keys.append(s3_key)
                    print(f"    ✓ Deleted: {s3_key}")
                except ClientError as e:
                    print(f"    ⚠ Failed to delete {s3_key}: {e.response['Error']['Message']}")
                except Exception as e:
                    print(f"    ⚠ Failed to delete {s3_key}: {str(e)}")
        elif delete_s3_images and s3_client is None:
            print(f"  ⚠ S3 client not configured, skipping S3 image deletion")
        
        # Delete full PDF files from S3 (always delete if S3 is configured)
        pdf_deleted_count = 0
        pdf_deleted_keys = []
        if s3_client is not None:
            print(f"  🗑️  Deleting full PDF files from S3...")
            try:
                # List all objects in the pdfs/ prefix
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix='pdfs/')
                
                # Create variations of pdf_name to match against
                # pdf_name is the stem (filename without extension) from ChromaDB
                # We need to match against the sanitized filename in S3
                pdf_name_lower = pdf_name.lower()
                
                # Find and delete matching PDFs
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            s3_key = obj['Key']
                            
                            # Extract filename from S3 key (format: pdfs/{timestamp}_{filename})
                            # Remove 'pdfs/' prefix and timestamp prefix
                            if s3_key.startswith('pdfs/'):
                                filename_part = s3_key[5:]  # Remove 'pdfs/'
                                # Remove timestamp prefix (format: YYYYMMDD_HHMMSS_)
                                # Find first underscore after potential timestamp
                                parts = filename_part.split('_', 2)
                                if len(parts) >= 3:
                                    # Likely has timestamp, get the filename part
                                    potential_filename = '_'.join(parts[2:])
                                else:
                                    # No timestamp or different format, use whole thing
                                    potential_filename = filename_part
                                
                                # Get filename without extension for comparison
                                potential_name_stem = Path(potential_filename).stem.lower()
                                
                                # Check if this PDF matches our pdf_name
                                if pdf_name_lower == potential_name_stem or pdf_name_lower in potential_filename.lower():
                                    try:
                                        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                                        pdf_deleted_count += 1
                                        pdf_deleted_keys.append(s3_key)
                                        print(f"    ✓ Deleted full PDF: {s3_key}")
                                    except ClientError as e:
                                        print(f"    ⚠ Failed to delete {s3_key}: {e.response['Error']['Message']}")
                                    except Exception as e:
                                        print(f"    ⚠ Failed to delete {s3_key}: {str(e)}")
                
                if pdf_deleted_count == 0:
                    print(f"    ⚠ No matching full PDF files found in S3 for '{pdf_name}'")
            except ClientError as e:
                print(f"    ⚠ Error listing/deleting PDFs from S3: {e.response['Error']['Message']}")
            except Exception as e:
                print(f"    ⚠ Error deleting PDFs from S3: {str(e)}")
        else:
            print(f"  ⚠ S3 client not configured, skipping full PDF deletion")
        
        # Trigger backup after deletion
        if s3_client is not None:
            print(f"  📤 Backing up ChromaDB after deletion...")
            def delayed_backup():
                time.sleep(2)  # Wait for ChromaDB to flush writes
                upload_chromadb_to_s3()
            threading.Thread(target=delayed_backup, daemon=True).start()
        
        print(f"\n{'='*80}")
        print(f"✅ PDF deletion completed")
        print(f"   PDF: {pdf_name}")
        print(f"   Pages deleted from ChromaDB: {page_count}")
        if delete_s3_images:
            print(f"   S3 images deleted: {s3_deleted_count}")
        print(f"   Full PDFs deleted from S3: {pdf_deleted_count}")
        print(f"{'='*80}\n")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"PDF '{pdf_name}' deleted successfully",
            "pdf_name": pdf_name,
            "pages_deleted_from_chromadb": page_count,
            "s3_images_deleted": s3_deleted_count if delete_s3_images else None,
            "s3_image_keys_deleted": s3_deleted_keys if delete_s3_images else None,
            "full_pdfs_deleted_from_s3": pdf_deleted_count,
            "full_pdf_keys_deleted": pdf_deleted_keys
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"  ❌ Error deleting PDF: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Failed to delete PDF: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Check for required API keys and credentials
    print("\n" + "="*80)
    print("🔧 Configuration Check")
    print("="*80)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not found!")
    else:
        print("✓ OpenAI API key configured")
    
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("⚠️  WARNING: AWS credentials not found!")
    else:
        print("✓ AWS credentials configured")
    
    if not os.getenv("TWILIO_ACCOUNT_SID") or not os.getenv("TWILIO_AUTH_TOKEN"):
        print("⚠️  WARNING: Twilio credentials not found!")
    else:
        print("✓ Twilio credentials configured")
    
    if not os.getenv("SMTP_USERNAME") or not os.getenv("SMTP_PASSWORD"):
        print("⚠️  WARNING: SMTP credentials not found!")
    else:
        print("✓ SMTP credentials configured")
    
    if not GOOGLE_CALENDAR_AVAILABLE:
        print("⚠️  WARNING: Google Calendar libraries not available!")
    elif not os.getenv("GOOGLE_CALENDAR_CREDENTIALS"):
        print("⚠️  WARNING: GOOGLE_CALENDAR_CREDENTIALS not found!")
    else:
        print("✓ Google Calendar credentials configured")
    
    print("\nPlease ensure your .env file contains:")
    print("  OPENAI_API_KEY=your-openai-api-key")
    print("  AWS_ACCESS_KEY_ID=your-aws-access-key")
    print("  AWS_SECRET_ACCESS_KEY=your-aws-secret-key")
    print("  AWS_REGION=us-east-1")
    print("  S3_BUCKET_NAME=your-bucket-name")
    print("  GOOGLE_CALENDAR_CREDENTIALS=your-google-calendar-credentials-json")
    print("  GOOGLE_CALENDAR_TOKEN=base64-encoded-token-pickle (optional, for pre-authenticated tokens)")
    print("="*80 + "\n")
    
    # Use PORT environment variable (Cloud Run sets this), default to 8080
    port = int(os.getenv("PORT", "8080"))
    
    # Start ngrok tunnel in background thread (non-blocking)
    def start_ngrok_background():
        """Start ngrok tunnel in background after a short delay"""
        time.sleep(2)  # Wait for server to start
        try:
            public_url = start_ngrok_tunnel(port)
            if public_url:
                print(f"🎉 Service is accessible via ngrok at: {public_url}")
        except Exception as e:
            print(f"⚠️  Failed to start ngrok tunnel: {e}")
            print("⚠️  Continuing without ngrok. Service will only be accessible locally.")
    
    # Start ngrok in background thread
    if PYNGROK_AVAILABLE:
        threading.Thread(target=start_ngrok_background, daemon=True).start()
    
    print(f"🚀 Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
