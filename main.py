import fitz  # PyMuPDF
import google.generativeai as genai
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import io
import uuid
import os
import smtplib
import imaplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import markdown
from xhtml2pdf import pisa
import re
import logging
import sys
import asyncio
from typing import Tuple, Dict, Optional, List
from functools import partial
from pydantic import BaseModel


from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager



import sqlite3
from datetime import datetime
import mysql.connector 
from mysql.connector import Error 
# --- Load Environment Variables ---
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("WARNING: BeautifulSoup4 is not installed. The email processing endpoint will not work.")
    print("Please run: pip install beautifulsoup4")
    BeautifulSoup = None



# ### MODIFIED FOR FILE LOGGING AND REPORT STORAGE ###
REPORTS_DIR = 'analysis_reports' # Define a directory for saved reports

# ### DB INTEGRATION ###
# DATABASE_FILE = 'recruitment.db'
DATABASE_FILE = 'database/recruitment.db' # The new path inside the container

CV_UPLOADS_DIR = 'uploaded_cvs'   # NEW: Define a directory for uploaded CVs

# Create logs, reports, and CV directories if they don't exist
if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    print(f"Created directory for saving reports: {REPORTS_DIR}")
if not os.path.exists(CV_UPLOADS_DIR):
    os.makedirs(CV_UPLOADS_DIR)
    print(f"Created directory for saving CVs: {CV_UPLOADS_DIR}")
if not os.path.exists(DATABASE_FILE):
    os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
    print(f"Created directory for saving database: {os.path.dirname(DATABASE_FILE)}")



logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logging.info("Application starting up... (Logging to file: logs/app.log)")

# --- Configuration ---
# Google AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# SMTP Configuration for Emailing Reports
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_SENDER_EMAIL = os.getenv("SMTP_SENDER_EMAIL")
SMTP_SENDER_PASSWORD = os.getenv("SMTP_SENDER_PASSWORD")



# IMAP Configuration for Reading Emails
IMAP_SERVER = os.getenv("IMAP_SERVER")
IMAP_EMAIL = os.getenv("IMAP_EMAIL")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD")
RECIPIENT_EMAILS_STR = os.getenv("RECIPIENT_EMAILS", "")
RECIPIENT_EMAILS = [email.strip() for email in RECIPIENT_EMAILS_STR.split(',') if email.strip()]

# --- Log the loaded configuration ---
logging.info(f"SMTP Server: '{SMTP_SERVER}', Port: {SMTP_PORT}, Sender: '{SMTP_SENDER_EMAIL}'")
logging.info(f"IMAP Server: '{IMAP_SERVER}', User: '{IMAP_EMAIL}'")
logging.info(f"Default Report Recipients: {RECIPIENT_EMAILS}")
logging.info(f"Analysis reports will be saved to: '{REPORTS_DIR}'") # Log the reports directory
logging.info(f"Database file located at: '{DATABASE_FILE}'") # Log the DB file

logging.info(f"Uploaded CVs will be saved to: '{CV_UPLOADS_DIR}'") # Log the CV uploads directory


# --- Scheduler Initialization ---
scheduler = AsyncIOScheduler()


def init_db():
    """
    Initializes the SQLite database and creates the candidates table if it doesn't exist.
    The table now includes 'cv_filepath' and 'report_filepath' to store paths to the
    candidate's resume and the generated analysis report.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        #  Added cv_filepath and report_filepath columns ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                position TEXT NOT NULL,
                score INTEGER NOT NULL,
                application_date TEXT NOT NULL,
                cv_filepath TEXT,
                report_filepath TEXT,
                UNIQUE(name, position)
            )
        """)
        conn.commit()
        conn.close()
        logging.info("Database initialized successfully with 'cv_filepath', 'report_filepath' columns and UNIQUE constraint.")
    except Exception as e:
        logging.error("Failed to initialize database", exc_info=True)


# --- Lifespan Manager for Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup and shutdown events.
    Starts the scheduler on startup and shuts it down gracefully on exit.
    """
    logging.info("Application startup: Initializing database...")
    init_db() # ### DB INTEGRATION ###
    logging.info("Application startup: Starting scheduler...")
    scheduler.start()
    yield  # The application is now running
    logging.info("Application shutdown: Shutting down scheduler...")
    scheduler.shutdown()

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static") # ADD THIS LINE

templates = Jinja2Templates(directory="templates")

# --- In-memory Cache for Web UI Analyses ---
analysis_cache = {}

# --- Load Google Gemini Model (once on startup) ---
model = None
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY not found in environment variables. AI features will be disabled.")
else:
    try:
        logging.info("Configuring Google Gemini model...")
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        logging.info("Google Gemini model configured successfully.")
    except Exception as e:
        logging.error("Error configuring Google Gemini model", exc_info=True)

# --- Helper Functions ---
def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extracts text from a PDF file's byte content."""
    text = ""
    try:
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logging.error(f"Could not read the PDF file. Details: {e}", exc_info=True)
        return ""
    return text

def _get_name_from_ai(resume_text: str, model) -> Optional[str]:
    """Internal helper to ask the AI model to extract the name as a fallback."""
    if not model:
        logging.warning("AI model not available for name extraction.")
        return None
    try:
        prompt = f"From the following resume text, extract only the full name of the candidate. Do not add any other words or explanation. Just return the name.\n\nResume Text:\n\"\"\"\n{resume_text[:2000]}\n\"\"\""
        response = model.generate_content(prompt, generation_config={"temperature": 0.0,"top_p": 1,"top_k": 1})
        name = response.text.strip()
        if ' ' in name and len(name.split()) < 6 and '\n' not in name:
            logging.info(f"AI successfully extracted name: '{name}'")
            return name
        else:
            logging.warning(f"AI returned an unlikely name: '{name}'. Discarding.")
            return None
    except Exception as e:
        logging.error(f"AI name extraction failed.", exc_info=True)
    return None

def extract_candidate_name(resume_text: str, model) -> str:
    """
    Extracts a candidate's name from resume text using a multi-layered approach.
    1. Looks for explicit "Name:" or "Applicant:" labels.
    2. Uses heuristics to find a likely name (Title Case or ALL CAPS) in the first few lines.
    3. As a fallback, asks the AI model to identify the name.
    4. If all else fails, returns a default string.
    """
    # First, check for an explicit "Name:" field
    name_match = re.search(r"^(?:Name|Applicant)\s*[:\-\s]\s*(.+)$", resume_text, re.IGNORECASE | re.MULTILINE)
    if name_match:
        # Take only the first line of the matched value
        return name_match.group(1).strip().split('\n')[0]

    # --- MODIFIED HEURISTICS ---
    # Keywords that strongly suggest a line is a job title, not a name.
    job_title_keywords = [
        'engineer', 'developer', 'manager', 'analyst', 'specialist', 'consultant',
        'architect', 'designer', 'scientist', 'lead', 'head', 'director',
        'president', 'officer', 'executive', 'coordinator', 'assistant', 'associate',
        'presales', 'technical', 'recruitment', 'solutions'
    ]

    # Heuristic search in the first few lines of the resume
    for line in resume_text.split('\n')[:10]:
        line = line.strip()
        # Basic filters: ignore empty lines, long lines, or lines with contact info
        if not line or len(line) > 50: continue
        if '@' in line or re.search(r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})', line) or "linkedin.com" in line or "github.com" in line:
            continue

        # Filter out lines containing job title keywords
        if any(keyword in line.lower() for keyword in job_title_keywords):
            continue

        # Regex for a plausible name (2-4 words, Title Case or ALL CAPS)
        name_match = re.match(r"^([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){1,3})$", line) # Title Case
        if not name_match:
             name_match = re.match(r"^([A-Z][A-Z'-]+(?:\s+[A-Z][A-Z'-]+){1,3})$", line) # ALL CAPS

        if name_match:
            return name_match.group(1)

    # If heuristics fail, fall back to the AI model
    logging.warning("Regex patterns failed to find candidate name. Attempting AI extraction.")
    ai_name = _get_name_from_ai(resume_text, model)
    if ai_name:
        return ai_name

    logging.error("All methods failed to extract a candidate name from the resume.")
    return "Unknown Candidate"

def extract_job_vacancy(job_description_text: str) -> str:
    """Extracts the job vacancy/title from the job description text."""
    patterns = [
        r"Job\s*Title\s*:\s*(.+)", r"Position\s*:\s*(.+)",
        r"Vacancy\s*:\s*(.+)", r"Role\s*:\s*(.+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, job_description_text, re.IGNORECASE)
        if match:
            # Return the first line of the match to handle multi-line values
            return match.group(1).strip().split('\n')[0]

    # --- MODIFIED FALLBACK LOGIC ---
    # List of common section headers to ignore.
    ignore_headers = [
        'responsibilities', 'requirements', 'qualifications', 'duties',
        'skills', 'experience', 'about the role', 'what you will do',
        'what we offer', 'benefits', 'summary', 'description'
    ]

    # Search the first few lines for a plausible title, ignoring headers.
    for line in job_description_text.split('\n')[:10]: # Check top 10 lines
        line = line.strip()
        # A good title is usually short, not a full sentence, and not a common header.
        if line and len(line) < 100 and not line.endswith(('.', ':')) and line.lower() not in ignore_headers:
             # Check if it's not just a generic phrase
             if len(line.split()) < 8: # Titles are usually not long phrases
                return line

    return "Not Specified"

def _blocking_save_candidate_to_db(name: str, position: str, score: int, cv_filepath: Optional[str], report_filepath: Optional[str]):
    """
    Synchronous function to save or update candidate data in SQLite, including paths
    to the CV and the analysis report.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        application_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # --- MODIFIED: Added report_filepath to INSERT and UPDATE ---
        upsert_sql = """
            INSERT INTO candidates (name, position, score, application_date, cv_filepath, report_filepath)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name, position) DO UPDATE SET
                score = excluded.score,
                application_date = excluded.application_date,
                cv_filepath = excluded.cv_filepath,
                report_filepath = excluded.report_filepath;
        """
        cursor.execute(upsert_sql, (name, position, score, application_date, cv_filepath, report_filepath))
        if cursor.lastrowid:
             logging.info(f"Successfully INSERTED new candidate '{name}' for position '{position}' with CV and report paths.")
        else:
             logging.info(f"Successfully UPDATED existing candidate '{name}' for position '{position}' with new paths.")
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to save or update candidate '{name}' to database.", exc_info=True)

async def save_candidate_to_db(name: str, position: str, score: int, cv_filepath: Optional[str], report_filepath: Optional[str]):
    """Asynchronously saves candidate data by running the blocking DB code in a thread pool."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        partial(_blocking_save_candidate_to_db, name, position, score, cv_filepath, report_filepath)
    )


def _blocking_delete_candidate_from_db(candidate_id: int) -> bool:
    """
    Synchronous function to delete a candidate from SQLite by ID.
    Returns True if a row was deleted, False otherwise.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
        deleted_rows = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted_rows > 0:
            logging.info(f"Successfully deleted candidate with ID: {candidate_id}")
            return True
        else:
            logging.warning(f"Attempted to delete non-existent candidate with ID: {candidate_id}")
            return False
    except Exception as e:
        logging.error(f"Failed to delete candidate with ID {candidate_id} from database.", exc_info=True)
        raise

async def delete_candidate_from_db(candidate_id: int):
    """Asynchronously deletes a candidate by running the blocking DB code in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        partial(_blocking_delete_candidate_from_db, candidate_id)
    )

def analyze_resume_with_ai(resume_text: str, job_description: str) -> str:
    """Sends resume and job description to Gemini for analysis."""
    if not model:
        raise HTTPException(status_code=503, detail="Gemini model is not available.")
    
    generation_config = {
        "temperature": 0.0, "top_p": 1, "top_k": 1, "max_output_tokens": 8192,
    }

    system_prompt = (
        "You are an expert recruitment assistant. Your goal is to provide a deterministic and consistent assessment of a candidate's resume against a job description. "
        "Provide your response in Markdown format using the following structured sections:\n\n"
        "### 1. 🧾 Resume Summary\n"
        "- A concise summary of the candidate's background, key experiences, and skills.\n\n"
        "### 2. ✅ Match Evaluation Against Job Role\n"
        "- Compare the resume to the job description point-by-point. Use bullet points.\n"
        "- Highlight strong matches and potential gaps.\n\n"
        "### 3. 🧠 Final Assessment\n"
        "- A final judgment on the candidate's suitability.\n"
        "- A recommendation on whether to shortlist for an interview.\n"
        "- Suggest specific questions to ask during the interview.\n\n"
        "### 4. 🎯 Match Score\n"
        "- Provide a numerical score from 0 to 100 representing the candidate's overall match. The score must be on its own line in the format 'Score: [number]'."
    )
    user_prompt = f"Based on the instructions provided, please analyze the following resume against the job description.\n\n**Resume Text:**\n```\n{resume_text}\n```\n\n**Job Description:**\n```\n{job_description}\n```"
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        response = model.generate_content(full_prompt,generation_config=generation_config)
        return response.text
    except Exception as e:
        logging.error(f"Error during Gemini API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error communicating with AI model: {e}")

def create_pdf_from_markdown(markdown_content: str, resume_filename: str, applicant_details: Optional[Dict[str, str]] = None) -> io.BytesIO:
    """Creates a styled PDF from markdown, optionally including a table of applicant details."""
    details_html = ""
    if applicant_details:
        details_html += "<h2>Applicant Details</h2><table>"
        for key, value in applicant_details.items():
            details_html += f"<tr><td style='padding-right: 15px;'><strong>{key}</strong></td><td>{value}</td></tr>"
        details_html += "</table><hr>"

    analysis_html = markdown.markdown(markdown_content)
    styled_html = f"""
    <html><head><style>
        body {{ font-family: sans-serif; line-height: 1.6; color: #333; }}
        h1, h2, h3 {{ color: #000; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 20px;}}
        h1 {{ font-size: 24px; }} h2 {{ font-size: 20px; }}
        table {{ border-collapse: collapse; margin-bottom: 20px; }}
        td {{ padding: 5px; vertical-align: top; }}
        ul {{ padding-left: 20px; }} strong {{ color: #000; }}
        hr {{ border: 0; border-top: 1px solid #ccc; }}
    </style></head><body>
        <h1>Resume Analysis Report</h1>
        <p><em>Based on resume: {resume_filename}</em></p>
        <hr>
        {details_html}
        <h2>AI-Powered Analysis</h2>
        {analysis_html}
    </body></html>"""
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(styled_html), dest=pdf_buffer)
    if pisa_status.err:
        logging.error("PDF generation failed.")
        raise Exception("PDF generation failed.")
    pdf_buffer.seek(0)
    return pdf_buffer

def parse_email_body(body_text: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Parses the text from an email body to extract applicant details and the job description.
    This version is enhanced to handle multi-line keys (e.g., 'Position' on one line
    and 'Applied: ...' on the next).
    """
    applicant_details = {}
    job_description = None
    details_part = None
    try:
        # Find a separator between applicant details and the job description.
        # A line with only a colon is a common pattern.
        if '\n:\n' in body_text:
            details_part, job_description = body_text.split('\n:\n', 1)
        # Fallback for other potential separators if needed
        elif "We require a talented" in body_text:
            details_part, job_desc_part = body_text.split("We require a talented", 1)
            job_description = "We require a talented" + job_desc_part
        else:
            # If no clear separator, we might have to guess, but for now, we'll raise an error.
            raise ValueError("Could not find a known separator (e.g., a line with only ':') between applicant details and job description.")

        details_part = details_part.strip()
        job_description = job_description.strip()

        # --- NEW LOGIC to handle multi-line keys ---
        lines = details_part.split('\n')
        processed_lines = []
        i = 0
        while i < len(lines):
            current_line = lines[i].strip()
            # Check if the current line looks like a key fragment (no colon) and the next line completes it
            if ':' not in current_line and i + 1 < len(lines) and ':' in lines[i+1]:
                # Combine this line with the next one to form a single key-value line
                next_line = lines[i+1].strip()
                combined_line = f"{current_line} {next_line}"
                processed_lines.append(combined_line)
                i += 2  # We've processed two lines, so jump ahead
            else:
                processed_lines.append(current_line)
                i += 1
        # --- END OF NEW LOGIC ---

        # Process the (potentially combined) lines to extract key-value pairs
        for line in processed_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    applicant_details[key] = value

        return applicant_details, job_description
    except Exception as e:
        logging.error(f"Could not parse email body. Error: {e}. Check email format.", exc_info=True)
        return None, None

def _blocking_send_email(subject: str, body: str, recipients: List[str], attachment_buffer: io.BytesIO, filename: str):
    """Synchronous email sending logic to be run in a thread pool."""
    if not all([SMTP_SERVER, SMTP_SENDER_EMAIL, SMTP_SENDER_PASSWORD]):
        logging.error("Email sending failed: SMTP service is not configured.")
        raise ValueError("SMTP service is not configured on the server.")

    msg = MIMEMultipart()
    msg['From'] = SMTP_SENDER_EMAIL
    msg['To'] = ", ".join(recipients)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment_buffer.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename={filename}")
    msg.attach(part)

    try:
        logging.info(f"Connecting to SMTP server {SMTP_SERVER}:{SMTP_PORT}...")
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                server.login(SMTP_SENDER_EMAIL, SMTP_SENDER_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_SENDER_EMAIL, SMTP_SENDER_PASSWORD)
                server.send_message(msg)
        logging.info(f"Email sent successfully to: {', '.join(recipients)}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}", exc_info=True)
        raise

async def send_email_with_attachment(subject: str, body: str, recipients: List[str], attachment_buffer: io.BytesIO, filename: str):
    """Asynchronously sends an email by running the blocking smtplib code in a thread pool."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        partial(_blocking_send_email, subject, body, recipients, attachment_buffer, filename)
    )
# ... after send_email_with_attachment function ...

def extract_email_from_text(text: str) -> Optional[str]:
    """Extracts the first found email address from a block of text."""
    # This regex is widely used for email validation
    match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    if match:
        return match.group(0)
    return None

def get_email_template(position: str, candidate_name: str) -> Tuple[str, str]:
    """Selects an email template based on the job position."""
    position_lower = position.lower()
    
    # ▼▼▼ CHECK FOR SALES/NON-TECHNICAL ROLES FIRST ▼▼▼
    if any(keyword in position_lower for keyword in ['sales', 'account manager']):
        subject = f"Next Steps: Your Application for the {position} role at UNIQUE COMPUTER SYSTEMS LLC"
        body = f"""Dear {candidate_name},

Thank you for your interest in UNIQUE COMPUTER SYSTEMS LLC. We have reviewed your application for the {position} position and are pleased to inform you that you have been shortlisted to move forward in our evaluation process.

The next step in our process is an Initial Interview (HR Evaluation call).

Guidance for Your Next Step: HR Evaluation Call
• Purpose: This initial 30-minute call with our HR team is to learn more about your professional background, assess your communication skills, and discuss your alignment with the role.
• What to Expect: A member of our HR team will be contacting you via email or phone within the next 2-3 business days to schedule a time that is convenient for you.

We were impressed with your background and look forward to speaking with you soon.

Best regards,
The HR Team
UNIQUE COMPUTER SYSTEMS LLC"""
        return subject, body

    # ▼▼▼ THEN CHECK FOR TECHNICAL ROLES ▼▼▼
    if any(keyword in position_lower for keyword in ['developer', 'architect', 'engineer']):
        subject = f"Next Steps: Your Application for the {position} role at UNIQUE COMPUTER SYSTEMS LLC"
        body = f"""Dear {candidate_name},

Thank you for your interest in UNIQUE COMPUTER SYSTEMS LLC. We have reviewed your application for the {position} position and are pleased to inform you that you have been shortlisted to move forward in our evaluation process.

As per our hiring process for technical roles, the next step is an online technical assessment designed to evaluate your core competencies.

Guidance for Your Next Step: Online Assessment
• Test Link: [Insert Technical Test Link Here]
• Duration: 1 Hour
• Focus: This assessment will cover key areas relevant to the {position} position. Please ensure you are in a quiet environment with a stable internet connection.

Upon successful completion of this assessment, a member of our HR team will contact you to discuss the subsequent stages.

We were impressed with your background and look forward to learning more about you.

Best regards,
The HR Team
UNIQUE COMPUTER SYSTEMS LLC"""
        return subject, body

    # Default Fallback Template (remains the same)
    subject = f"Your Application for {position} at UNIQUE COMPUTER SYSTEMS LLC"
    body = f"""Dear {candidate_name},

Thank you for applying for the {position} role. We have reviewed your application and would like to invite you to the next step in our process.

Our HR team will be in touch shortly with more details.

Best regards,
The HR Team
UNIQUE COMPUTER SYSTEMS LLC"""
    return subject, body

def _blocking_send_plain_text_email(subject: str, body: str, recipient: str):
    """Synchronous plain text email sending logic."""
    if not all([SMTP_SERVER, SMTP_SENDER_EMAIL, SMTP_SENDER_PASSWORD]):
        raise ValueError("SMTP service is not configured.")

    msg = MIMEMultipart()
    msg['From'] = SMTP_SENDER_EMAIL
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                server.login(SMTP_SENDER_EMAIL, SMTP_SENDER_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_SENDER_EMAIL, SMTP_SENDER_PASSWORD)
                server.send_message(msg)
        logging.info(f"Plain text email sent successfully to: {recipient}")
    except Exception as e:
        logging.error(f"Failed to send plain text email: {e}", exc_info=True)
        raise

async def send_plain_text_email(subject: str, body: str, recipient: str):
    """Asynchronously sends a plain text email."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        partial(_blocking_send_plain_text_email, subject, body, recipient)
    )
# --- API Endpoints (Web UI) ---
# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     """Serves the main HTML page for manual uploads."""
#     return templates.TemplateResponse("index.html", {"request": request})
# main.py

# --- API Endpoints (Web UI) ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page and populates it with vacancies from MySQL."""
    vacancies = []
    try:
        # Only attempt to connect if all MySQL config variables are present
        if all([MYSQL_HOST, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD]):
            logging.info("Connecting to MySQL to fetch vacancies...")
            conn = mysql.connector.connect(
                host=MYSQL_HOST,
                database=MYSQL_DATABASE,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD
            )

            if conn.is_connected():
                cursor = conn.cursor(dictionary=True) # dictionary=True makes results easier to use
                # As seen in your screenshot, fetching active careers
                # query = "SELECT subject,bdesc,description FROM tblcareers WHERE isActive = 'y' ORDER BY subject"
                query = "SELECT subject, bdesc, jobcode, description FROM tblcareers WHERE isActive = 'y' ORDER BY subject"
                cursor.execute(query)
                vacancies = cursor.fetchall()
                cursor.close()
                conn.close()
                logging.info(f"Successfully fetched {len(vacancies)} active vacancies from MySQL.")
        else:
            logging.warning("MySQL environment variables not set. Skipping vacancy fetching.")

    except Error as e:
        logging.error(f"Failed to fetch vacancies from MySQL: {e}", exc_info=True)
        vacancies = [] # Ensure vacancies is an empty list on error

    return templates.TemplateResponse("index.html", {"request": request, "vacancies": vacancies})




@app.get("/candidates", response_class=HTMLResponse)
async def view_candidates(request: Request):
    """Serves the page to view all saved candidates from the database."""
    candidates_list = []
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # --- MODIFIED: Select cv_filepath and report_filepath as well ---
        cursor.execute("SELECT id, name, position, score, application_date, cv_filepath, report_filepath FROM candidates ORDER BY application_date DESC")
        rows = cursor.fetchall()
        conn.close()
        for row in rows:
            candidates_list.append(dict(row))
        logging.info(f"Fetched {len(candidates_list)} candidates from the database.")
    except Exception as e:
        logging.error("Failed to fetch candidates from database.", exc_info=True)
        return templates.TemplateResponse("candidates.html", {"request": request, "candidates": [], "error": str(e)})

    return templates.TemplateResponse("candidates.html", {"request": request, "candidates": candidates_list})


@app.delete("/candidates/{candidate_id}", status_code=204)
async def delete_candidate(candidate_id: int):
    """Deletes a candidate record from the database."""
    try:
        success = await delete_candidate_from_db(candidate_id)
        if not success:
            raise HTTPException(status_code=404, detail="Candidate not found")
        return
    except Exception as e:
        logging.error(f"Error deleting candidate {candidate_id}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not delete candidate: {str(e)}")


@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    """Analyzes a resume uploaded via the web form."""
    pdf_content = await resume.read()
    
    # --- MODIFIED: Capture the save_path for the CV for the database ---
    cv_save_path = None
    try:
        sanitized_original_filename = re.sub(r'[^a-zA-Z0-9_.-]', '', resume.filename)
        unique_filename = f"{uuid.uuid4()}_{sanitized_original_filename}"
        cv_save_path = os.path.join(CV_UPLOADS_DIR, unique_filename)
        with open(cv_save_path, "wb") as f:
            f.write(pdf_content)
        logging.info(f"Successfully saved uploaded CV to: {cv_save_path}")
    except Exception as e:
        logging.error(f"Could not save the uploaded CV file '{resume.filename}'. Error: {e}", exc_info=True)
    # --- END OF MODIFICATION ---

    resume_text = extract_text_from_pdf(pdf_content)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the uploaded PDF.")

    try:
        candidate_name = extract_candidate_name(resume_text, model)
        if candidate_name == "Unknown Candidate":
            candidate_name = f"Candidate from {resume.filename}"

        job_vacancy = extract_job_vacancy(job_description)
        applicant_details = {
            "Candidate Name": candidate_name,
            "Position Applied For": job_vacancy
        }
        logging.info(f"Extracted details for web upload: {applicant_details}")

        analysis_result = analyze_resume_with_ai(resume_text, job_description)

        chat_history = [
            {'role': 'user', 'parts': [f"Analyze this resume:\n{resume_text}\n\nAgainst this job description:\n{job_description}"]},
            {'role': 'model', 'parts': [analysis_result]}
        ]

        match_rate = 0
        score_match = re.search(r"Score:\s*(\d+)", analysis_result, re.IGNORECASE)
        if score_match:
            match_rate = int(score_match.group(1))

        pdf_buffer = create_pdf_from_markdown(analysis_result, resume.filename, applicant_details)
        pdf_bytes = pdf_buffer.getvalue()

        # --- MODIFIED: Save report and get its path for the database ---
        report_filepath = None
        try:
            sanitized_filename = re.sub(r'[^a-zA-Z0-9_.-]', '', resume.filename)
            report_filename = f"Analysis-{sanitized_filename}"
            report_filepath = os.path.join(REPORTS_DIR, report_filename)
            with open(report_filepath, "wb") as f:
                f.write(pdf_bytes)
            logging.info(f"Successfully saved analysis report to {report_filepath}")
        except Exception as e:
            logging.error(f"Failed to save report to {report_filepath}", exc_info=True)
        
        # --- MODIFIED: Pass both CV and report paths to the database function ---
        await save_candidate_to_db(candidate_name, job_vacancy, match_rate, cv_save_path, report_filepath)

        analysis_id = str(uuid.uuid4())
        analysis_cache[analysis_id] = {
            "markdown": analysis_result,
            "pdf": pdf_bytes,
            "chat_history": chat_history,
            "filename": resume.filename
        }

        return JSONResponse(content={
            "analysis": analysis_result,
            "analysis_id": analysis_id,
            "match_rate": match_rate
        })
    except Exception as e:
        logging.error("Error during manual analysis", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/chat")
async def chat_with_analyzer(request: Request):
    """Handles follow-up questions for a specific analysis."""
    if not model:
        raise HTTPException(status_code=503, detail="Gemini model is not available.")

    data = await request.json()
    analysis_id = data.get("analysis_id")
    prompt = data.get("prompt")

    if not all([analysis_id, prompt]):
        raise HTTPException(status_code=400, detail="Analysis ID and prompt are required.")

    cached_data = analysis_cache.get(analysis_id)
    if not cached_data:
        raise HTTPException(status_code=404, detail="Analysis not found or expired.")

    try:
        chat = model.start_chat(history=cached_data["chat_history"])
        response = chat.send_message(prompt)
        response_text = response.text
        analysis_cache[analysis_id]["chat_history"] = chat.history
        return JSONResponse(content={"response": response_text})
    except Exception as e:
        logging.error("Error during chat inference", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gemini inference error: {e}")

@app.get("/download/{analysis_id}")
async def download_analysis_pdf(analysis_id: str):
    """Allows downloading of a cached analysis PDF."""
    cached_data = analysis_cache.get(analysis_id)
    if not cached_data:
        raise HTTPException(status_code=404, detail="Analysis not found or expired.")

    pdf_bytes = cached_data["pdf"]
    filename = f"Analysis-{cached_data['filename']}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type='application/pdf',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )

@app.post("/email")
async def email_analysis(request: Request):
    """Emails a cached analysis PDF to a list of recipients."""
    data = await request.json()
    analysis_id = data.get("analysis_id")
    recipients = data.get("recipients")

    if not all([analysis_id, recipients]):
        raise HTTPException(status_code=400, detail="Analysis ID and recipients are required.")

    cached_data = analysis_cache.get(analysis_id)
    if not cached_data:
        raise HTTPException(status_code=404, detail="Analysis not found or expired.")

    try:
        pdf_buffer = io.BytesIO(cached_data["pdf"])
        subject = f"Resume Analysis for {cached_data['filename']}"
        body = "Please find the attached resume analysis report."
        filename = f"Analysis-{cached_data['filename']}.pdf"
        await send_email_with_attachment(subject, body, recipients, pdf_buffer, filename)
        return JSONResponse(content={"message": "Email sent successfully!"})
    except Exception as e:
        logging.error(f"Email sending failed for analysis ID {analysis_id}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")
# ... after the /email endpoint ...

class CandidateEmail(BaseModel):
    to_email: str
    subject: str
    body: str

@app.get("/get-email-details/{candidate_id}")
async def get_email_details(candidate_id: int):
    """
    Fetches candidate details, extracts email from CV, and generates an email template.
    """
    cv_path = await asyncio.get_running_loop().run_in_executor(None, partial(_blocking_get_cv_path, candidate_id))
    if not cv_path or not os.path.exists(cv_path):
        raise HTTPException(status_code=404, detail="CV file not found for this candidate.")

    try:
        # Get other candidate details from DB
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT name, position FROM candidates WHERE id = ?", (candidate_id,))
        candidate_data = cursor.fetchone()
        conn.close()
        if not candidate_data:
            raise HTTPException(status_code=404, detail="Candidate record not found.")

        # Extract text and email from CV
        with open(cv_path, "rb") as f:
            pdf_content = f.read()
        
        resume_text = extract_text_from_pdf(pdf_content)
        candidate_email = extract_email_from_text(resume_text)

        # Generate template
        subject, body = get_email_template(candidate_data['position'], candidate_data['name'])

        return JSONResponse(content={
            "email": candidate_email,
            "subject": subject,
            "body": body
        })

    except Exception as e:
        logging.error(f"Failed to get email details for candidate {candidate_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/send-candidate-email")
async def send_candidate_email(email_data: CandidateEmail):
    """
    Sends a customized email to a candidate.
    """
    try:
        await send_plain_text_email(
            subject=email_data.subject,
            body=email_data.body,
            recipient=email_data.to_email
        )
        return JSONResponse(content={"message": "Email sent successfully!"})
    except Exception as e:
        logging.error(f"Failed to send candidate email to {email_data.to_email}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")

# ... existing download-cv endpoint ...

def _blocking_get_cv_path(candidate_id: int) -> Optional[str]:
    """Synchronous function to retrieve the CV file path for a candidate from the database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT cv_filepath FROM candidates WHERE id = ?", (candidate_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result and result[0] else None
    except Exception as e:
        logging.error(f"Failed to get CV path for candidate ID {candidate_id}", exc_info=True)
        return None

def _blocking_get_report_path(candidate_id: int) -> Optional[str]:
    """Synchronous function to retrieve the report file path for a candidate from the database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT report_filepath FROM candidates WHERE id = ?", (candidate_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result and result[0] else None
    except Exception as e:
        logging.error(f"Failed to get report path for candidate ID {candidate_id}", exc_info=True)
        return None

@app.get("/download-cv/{candidate_id}")
async def download_cv(candidate_id: int):
    """Downloads the original CV for a given candidate ID."""
    loop = asyncio.get_running_loop()
    cv_path = await loop.run_in_executor(None, partial(_blocking_get_cv_path, candidate_id))

    if not cv_path:
        raise HTTPException(status_code=404, detail="CV record not found for this candidate.")
    if not os.path.exists(cv_path):
        logging.error(f"CV file not found on disk for candidate {candidate_id} at path: {cv_path}")
        raise HTTPException(status_code=404, detail="CV file not found on disk. It may have been moved or deleted.")

    original_filename = os.path.basename(cv_path)
    if '_' in original_filename and len(original_filename.split('_')[0]) == 36:
         original_filename = '_'.join(original_filename.split('_')[1:])

    return FileResponse(path=cv_path, media_type='application/pdf', filename=original_filename)

@app.get("/download-report/{candidate_id}")
async def download_report(candidate_id: int):
    """Downloads the analysis report for a given candidate ID."""
    loop = asyncio.get_running_loop()
    report_path = await loop.run_in_executor(None, partial(_blocking_get_report_path, candidate_id))

    if not report_path:
        raise HTTPException(status_code=404, detail="Analysis report record not found for this candidate.")
    if not os.path.exists(report_path):
        logging.error(f"Report file not found on disk for candidate {candidate_id} at path: {report_path}")
        raise HTTPException(status_code=404, detail="Report file not found on disk. It may have been moved or deleted.")

    return FileResponse(path=report_path, media_type='application/pdf', filename=os.path.basename(report_path))

def _blocking_email_check():
    """
    Synchronous logic for checking and processing an email.
    This function contains all the blocking imaplib calls.
    """
    logging.info(f"Connecting to IMAP server {IMAP_SERVER}...")
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(IMAP_EMAIL, IMAP_PASSWORD)
    mail.select('inbox')
    search_subject_prefix = "Job application received for"
    status, data = mail.search(None, 'UNSEEN', f'(SUBJECT "{search_subject_prefix}")')

    if status != 'OK' or not data[0]:
        msg = f"No new unread emails found with subject: '{search_subject_prefix}'."
        logging.info(msg)
        mail.logout()
        return True, msg

    latest_email_id = data[0].split()[-1]
    logging.info(f"Found new unread email with ID {latest_email_id.decode()}. Fetching content...")
    status, msg_data = mail.fetch(latest_email_id, '(RFC822)')
    if status != 'OK':
        mail.logout()
        raise imaplib.IMAP4.error("Failed to fetch email content.")

    msg = email.message_from_bytes(msg_data[0][1])
    email_body_text, resume_pdf_bytes, resume_filename = None, None, "resume.pdf"
    logging.info("Parsing email for body and attachments...")
    for part in msg.walk():
        if part.get_content_maintype() == 'multipart': continue
        content_disposition = str(part.get("Content-Disposition"))
        if "attachment" in content_disposition and part.get_filename() and part.get_filename().lower().endswith('.pdf'):
            if not resume_pdf_bytes:
                resume_pdf_bytes = part.get_payload(decode=True)
                resume_filename = part.get_filename()
                logging.info(f"Found PDF attachment: {resume_filename}")
        elif part.get_content_type() == "text/plain" and "attachment" not in content_disposition:
            if not email_body_text:
                email_body_text = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                logging.info("Found plain text body.")
        elif part.get_content_type() == "text/html" and "attachment" not in content_disposition:
            if not email_body_text:
                html_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html_body, 'html.parser')
                email_body_text = soup.get_text(separator='\n', strip=True)
                logging.info("Found and extracted text from HTML body.")

    if not resume_pdf_bytes or not email_body_text:
        mail.logout()
        raise ValueError("Email processing failed: Could not find both a PDF resume and a text body.")
    
    # --- MODIFIED: Capture the save path for the CV for the database ---
    email_cv_save_path = None
    try:
        sanitized_email_filename = re.sub(r'[^a-zA-Z0-9_.-]', '', resume_filename)
        unique_email_filename = f"{uuid.uuid4()}_{sanitized_email_filename}"
        email_cv_save_path = os.path.join(CV_UPLOADS_DIR, unique_email_filename)
        with open(email_cv_save_path, "wb") as f:
            f.write(resume_pdf_bytes)
        logging.info(f"Successfully saved CV from email attachment to: {email_cv_save_path}")
    except Exception as e:
        logging.error(f"Could not save CV from email attachment '{resume_filename}'. Error: {e}", exc_info=True)
    # --- END OF MODIFICATION ---

    applicant_details, job_description = parse_email_body(email_body_text)
    if not applicant_details or not job_description:
        mail.logout()
        raise ValueError("Could not parse applicant details and job description from email body.")

    resume_text = extract_text_from_pdf(resume_pdf_bytes)
    if not resume_text:
        mail.logout()
        raise ValueError(f"The attached PDF '{resume_filename}' appears to be empty or unreadable.")

    final_applicant_details = applicant_details.copy()
    
    # --- MODIFIED LOGIC TO PRIORITIZE NAME FROM EMAIL ---
    # 1. Attempt to get the name directly from the parsed email body.
    # The key is typically 'Name' as seen in the example.
    extracted_name = final_applicant_details.get('Name')

    # 2. If the name is not found in the email body (is None or empty), fall back to analyzing the resume.
    if not extracted_name or not extracted_name.strip():
        logging.warning("Name not found in email body details. Attempting to extract from resume PDF.")
        extracted_name = extract_candidate_name(resume_text, model)
    else:
        # The name was found in the email body. Log it and use it.
        logging.info(f"Successfully used name '{extracted_name.strip()}' from email body.")
        extracted_name = extracted_name.strip()

    # 3. As a final fallback, if no name could be found from email or resume, use the filename.
    if not extracted_name or extracted_name == "Unknown Candidate":
        logging.warning("Could not extract name from email or resume. Using filename as fallback.")
        extracted_name = f"Candidate from {resume_filename}"

    final_applicant_details['Candidate Name'] = extracted_name

    # --- MODIFIED LOGIC TO HANDLE INCONSISTENT KEYS ---
    # Prioritize the position from the structured details part of the email, checking multiple possible keys.
    position_from_details = (
        final_applicant_details.get('Position') or
        final_applicant_details.get('Applied') or
        final_applicant_details.get('Position Applied')
    )

    if position_from_details:
        final_applicant_details['Position Applied For'] = position_from_details.strip()
        logging.info(f"Successfully extracted position '{position_from_details.strip()}' from email details.")
    else:
        # Fallback to extracting from the job description text if not in details
        logging.warning("Position not found in email details (checked 'Position', 'Applied', 'Position Applied'), attempting to extract from job description text.")
        final_applicant_details['Position Applied For'] = extract_job_vacancy(job_description)

    # Clean up old/redundant keys to standardize on 'Candidate Name' and 'Position Applied For'
    if 'Name' in final_applicant_details:
        del final_applicant_details['Name']
    if 'Position' in final_applicant_details:
        del final_applicant_details['Position']
    if 'Applied' in final_applicant_details:
        del final_applicant_details['Applied']
    if 'Position Applied' in final_applicant_details:
         del final_applicant_details['Position Applied']
    logging.info(f"Final applicant details for report: {final_applicant_details}")

    logging.info(f"Analyzing resume for {final_applicant_details.get('Candidate Name', 'N/A')}...")
    analysis_markdown = analyze_resume_with_ai(resume_text, job_description)

    match_rate = 0
    score_match = re.search(r"Score:\s*(\d+)", analysis_markdown, re.IGNORECASE)
    if score_match:
        match_rate = int(score_match.group(1))
    
    logging.info("Generating analysis PDF report...")
    analysis_pdf_buffer = create_pdf_from_markdown(analysis_markdown, resume_filename, final_applicant_details)
    pdf_bytes = analysis_pdf_buffer.getvalue()

    applicant_name = final_applicant_details.get('Candidate Name', 'Candidate')
    report_subject = f"Resume Analysis for {applicant_name} - {final_applicant_details.get('Position Applied For', 'N/A')}"
    report_body = f"Please find the attached AI-generated analysis for the application from {applicant_name}."
    
    # --- MODIFIED: Save report, get its path, and save everything to DB ---
    sanitized_report_filename = re.sub(r'[^a-zA-Z0-9_.-]', '', resume_filename)
    report_filename = f"Analysis-{sanitized_report_filename}.pdf"
    report_filepath = os.path.join(REPORTS_DIR, report_filename)
    try:
        with open(report_filepath, "wb") as f:
            f.write(pdf_bytes)
        logging.info(f"Successfully saved analysis report to {report_filepath}")
    except Exception as e:
        logging.error(f"Failed to save report to {report_filepath}", exc_info=True)
        report_filepath = None # Set to None if saving fails

    _blocking_save_candidate_to_db(
        name=final_applicant_details.get('Candidate Name', 'N/A'),
        position=final_applicant_details.get('Position Applied For', 'N/A'),
        score=match_rate,
        cv_filepath=email_cv_save_path,
        report_filepath=report_filepath
    )
    # --- END OF MODIFICATION ---

    return {
        "success": True,
        "mail_connection": mail,
        "email_id": latest_email_id,
        "report_subject": report_subject,
        "report_body": report_body,
        "recipients": RECIPIENT_EMAILS,
        "attachment_buffer": io.BytesIO(pdf_bytes),
        "report_filename": report_filename,
        "applicant_name": applicant_name
    }


async def run_email_processing_pipeline() -> Tuple[bool, str]:
    """
    Connects to IMAP, processes the latest unread job application, and sends a report.
    """
    if not all([IMAP_SERVER, IMAP_EMAIL, IMAP_PASSWORD]):
        msg = "IMAP service is not configured on the server."
        logging.warning(f"Email processing skipped: {msg}")
        return False, msg
    if not RECIPIENT_EMAILS:
        msg = "No recipient emails are configured for reports."
        logging.warning(f"Email processing skipped: {msg}")
        return False, msg
    if not BeautifulSoup:
        msg = "Server is missing 'beautifulsoup4' library required for email processing."
        logging.error(msg)
        return False, msg

    processing_result = None
    try:
        loop = asyncio.get_running_loop()
        processing_result = await loop.run_in_executor(None, _blocking_email_check)

        if isinstance(processing_result, tuple):
            return processing_result

        await send_email_with_attachment(
            subject=processing_result["report_subject"],
            body=processing_result["report_body"],
            recipients=processing_result["recipients"],
            attachment_buffer=processing_result["attachment_buffer"],
            filename=processing_result["report_filename"]
        )

        logging.info(f"Process successful. Marking email {processing_result['email_id'].decode()} as read (\\Seen).")
        await loop.run_in_executor(
            None,
            partial(processing_result["mail_connection"].store, processing_result['email_id'], '+FLAGS', '\\Seen')
        )

        success_msg = f"Successfully processed and sent report for {processing_result['applicant_name']}."
        return True, success_msg

    except Exception as e:
        error_msg = f"An error occurred during the email processing pipeline: {e}"
        logging.error(error_msg, exc_info=True)
        return False, str(e)
    finally:
        if processing_result and isinstance(processing_result, dict) and processing_result.get("mail_connection"):
            mail_conn = processing_result["mail_connection"]
            if mail_conn.state != 'LOGOUT':
                logging.info("Closing IMAP connection.")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, mail_conn.logout)

@app.post("/process-incoming-email", summary="Process Latest Job Application from Inbox")
async def process_job_application_email():
    """
    Manually triggers the email processing pipeline.
    """
    logging.info("Manual trigger for email processing initiated.")
    success, message = await run_email_processing_pipeline()
    if success:
        return JSONResponse(content={"message": message})
    else:
        raise HTTPException(status_code=500, detail=f"Email processing failed: {message}")
    


# --- Scheduled Task Definition ---
async def scheduled_email_check():
    """
    A wrapper function for the scheduler to call the main email processing logic.
    """
    logging.info("Scheduler: Running periodic email check...")
    await run_email_processing_pipeline()
    logging.info("Scheduler: Email check finished.")

scheduler.add_job(scheduled_email_check, 'interval', minutes=15, id="email_processing_job")