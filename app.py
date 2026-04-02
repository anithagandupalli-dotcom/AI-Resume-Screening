import sqlite3
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd
import PyPDF2
import streamlit as st
import yagmail
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title="AI Resume Screening",
    page_icon=":briefcase:",
    layout="wide",
    initial_sidebar_state="expanded",
)


DB_PATH = "resume.db"
JOB_DESCRIPTION_PATH = "job_description.txt"
DATASET_GLOB = "*dataset*.csv"
CREDENTIALS_PATH = "credentials.json"

SKILLS_LIST = [
    "Python",
    "Machine Learning",
    "Deep Learning",
    "SQL",
    "Data Analysis",
    "Communication",
    "Java",
    "C++",
    "React",
    "NodeJS",
    "HTML",
    "CSS",
    "JavaScript",
    "TensorFlow",
    "Pandas",
    "Numpy",
]


def apply_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --bg-deep: #07111f;
            --panel: rgba(15, 23, 42, 0.78);
            --panel-strong: rgba(15, 23, 42, 0.92);
            --line: rgba(255, 255, 255, 0.08);
            --text-soft: #cbd5e1;
            --text-muted: #94a3b8;
            --teal: #14b8a6;
            --blue: #0ea5e9;
            --amber: #f59e0b;
            --rose: #fb7185;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(34, 197, 94, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(249, 115, 22, 0.14), transparent 24%),
                radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.18), transparent 30%),
                linear-gradient(145deg, #10213a 0%, #18324f 48%, #0f172a 100%);
            color: #f8fafc;
        }

        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(16, 33, 58, 0.98) 0%, rgba(15, 23, 42, 0.98) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(21, 37, 63, 0.95), rgba(24, 50, 79, 0.9));
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.4rem;
            color: #f8fafc;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #cbd5e1;
            max-width: 820px;
        }

        .section-heading {
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .section-eyebrow {
            display: inline-block;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-size: 0.72rem;
            font-weight: 800;
            color: #67e8f9;
            margin-bottom: 0.45rem;
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 800;
            color: #f8fafc;
        }

        .section-copy {
            color: #cbd5e1;
            font-size: 0.95rem;
            max-width: 760px;
        }

        .section-card {
            background: rgba(19, 36, 61, 0.86);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 20px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18);
        }

        .metric-card {
            background: linear-gradient(180deg, rgba(20, 37, 61, 0.98), rgba(28, 54, 86, 0.9));
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 20px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 14px 30px rgba(2, 6, 23, 0.25);
        }

        .metric-label {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-bottom: 0.2rem;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 2rem;
            font-weight: 800;
        }

        .metric-caption {
            color: #cbd5e1;
            font-size: 0.85rem;
        }

        .tag {
            display: inline-block;
            margin: 0.2rem 0.35rem 0.2rem 0;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(20, 184, 166, 0.14);
            color: #99f6e4;
            border: 1px solid rgba(45, 212, 191, 0.24);
            font-size: 0.84rem;
            font-weight: 600;
        }

        .info-banner {
            background: linear-gradient(90deg, rgba(20, 184, 166, 0.24), rgba(59, 130, 246, 0.18));
            border: 1px solid rgba(125, 211, 252, 0.24);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            color: #e2e8f0;
            margin-bottom: 1rem;
        }

        .legend-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 1rem 0 1.25rem 0;
        }

        .legend-card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(30, 41, 59, 0.86));
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            padding: 1rem;
            min-height: 120px;
        }

        .legend-title {
            font-size: 1rem;
            font-weight: 800;
            color: #f8fafc;
            margin: 0.55rem 0 0.3rem 0;
        }

        .legend-copy {
            color: #cbd5e1;
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 800;
            border: 1px solid transparent;
        }

        .status-pill.shortlisted {
            background: rgba(20, 184, 166, 0.14);
            color: #99f6e4;
            border-color: rgba(45, 212, 191, 0.28);
        }

        .status-pill.needs-review {
            background: rgba(245, 158, 11, 0.14);
            color: #fde68a;
            border-color: rgba(251, 191, 36, 0.24);
        }

        .status-pill.pending-review {
            background: rgba(148, 163, 184, 0.14);
            color: #e2e8f0;
            border-color: rgba(148, 163, 184, 0.22);
        }

        .status-dot {
            width: 0.45rem;
            height: 0.45rem;
            border-radius: 50%;
            background: currentColor;
        }

        .glass-callout {
            background: linear-gradient(135deg, rgba(8, 47, 73, 0.65), rgba(30, 41, 59, 0.82));
            border: 1px solid rgba(103, 232, 249, 0.16);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        .candidate-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 0.4rem;
        }

        .candidate-card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.9));
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 22px;
            padding: 1.15rem;
            box-shadow: 0 16px 30px rgba(2, 6, 23, 0.22);
        }

        .candidate-card.highlight {
            border-color: rgba(45, 212, 191, 0.28);
            box-shadow: 0 18px 34px rgba(20, 184, 166, 0.14);
        }

        .candidate-topline {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.8rem;
            margin-bottom: 0.8rem;
        }

        .candidate-name {
            font-size: 1.08rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.25rem;
        }

        .candidate-meta {
            color: #94a3b8;
            font-size: 0.84rem;
        }

        .candidate-score {
            min-width: 82px;
            text-align: right;
        }

        .candidate-score-value {
            font-size: 1.65rem;
            font-weight: 800;
            color: #f8fafc;
            line-height: 1;
        }

        .candidate-score-label {
            color: #94a3b8;
            font-size: 0.78rem;
            margin-top: 0.2rem;
        }

        .mini-label {
            color: #94a3b8;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 0.85rem 0 0.35rem 0;
        }

        .candidate-copy {
            color: #cbd5e1;
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .tag-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.35rem;
        }

        .tag.subtle {
            background: rgba(148, 163, 184, 0.12);
            color: #e2e8f0;
            border-color: rgba(148, 163, 184, 0.18);
        }

        @media (max-width: 900px) {
            .candidate-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 900px) {
            .legend-grid {
                grid-template-columns: 1fr;
            }
        }

        .stButton > button {
            border-radius: 12px;
            border: none;
            font-weight: 700;
            padding: 0.55rem 1rem;
            background: linear-gradient(90deg, #22c55e, #0ea5e9);
            color: white;
            box-shadow: 0 10px 24px rgba(14, 165, 233, 0.2);
        }

        .stDownloadButton > button {
            border-radius: 12px;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background: rgba(21, 37, 63, 0.84);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 18px;
        }

        .login-wrap {
            max-width: 480px;
            margin: 4rem auto 0 auto;
            background: linear-gradient(180deg, rgba(20, 37, 61, 0.96), rgba(24, 50, 79, 0.94));
            padding: 2rem;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            box-shadow: 0 25px 60px rgba(0, 0, 0, 0.3);
        }

        .login-note {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin: 1rem 0;
            color: #e2e8f0;
        }

        .small-muted {
            color: #94a3b8;
            font-size: 0.9rem;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {
            background: rgba(255, 255, 255, 0.08);
            color: #f8fafc;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            background: rgba(255, 255, 255, 0.08);
            color: #f8fafc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            name TEXT,
            score REAL,
            skills TEXT
        )
        """
    )

    existing_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(candidates)").fetchall()
    }
    optional_columns = {
        "missing_skills": "TEXT",
        "status": "TEXT",
        "uploaded_at": "TEXT",
        "email": "TEXT",
        "source_file": "TEXT",
    }

    for column_name, column_type in optional_columns.items():
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE candidates ADD COLUMN {column_name} {column_type}"
            )

    conn.commit()
    return conn


def load_job_description():
    with open(JOB_DESCRIPTION_PATH, "r", encoding="utf-8") as file:
        return file.read()


def save_job_description(content):
    with open(JOB_DESCRIPTION_PATH, "w", encoding="utf-8") as file:
        file.write(content.strip())


def extract_text(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def calculate_score(resume_text, job_description):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform([resume_text, job_description])
    return cosine_similarity(matrix)[0][1]


def extract_skills(resume_text):
    resume_lower = resume_text.lower()
    matched = [skill for skill in SKILLS_LIST if skill.lower() in resume_lower]
    missing = [skill for skill in SKILLS_LIST if skill not in matched]
    return matched, missing


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_credentials():
    credentials_file = Path(CREDENTIALS_PATH)
    if not credentials_file.exists():
        return None

    try:
        with credentials_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(data, dict):
        return None

    username = str(data.get("username", "")).strip()
    password_hash = str(data.get("password_hash", "")).strip()
    if not username or not password_hash:
        return None

    return {"username": username, "password_hash": password_hash}


def save_credentials(username, password):
    credentials = {
        "username": username.strip(),
        "password_hash": hash_password(password),
    }
    with open(CREDENTIALS_PATH, "w", encoding="utf-8") as file:
        json.dump(credentials, file, indent=2)


def credentials_exist():
    return load_credentials() is not None


def verify_login(username, password):
    credentials = load_credentials()
    if not credentials:
        return False
    return (
        username.strip() == credentials["username"]
        and hash_password(password) == credentials["password_hash"]
    )


def normalize_columns(df):
    normalized = df.copy()
    normalized.columns = [column.strip().lower().replace(" ", "_") for column in df.columns]
    return normalized


def find_local_dataset_files():
    return sorted(Path(".").glob(DATASET_GLOB))


def import_candidates_from_csv(conn, csv_path, job_description, shortlist_threshold=75):
    raw_df = pd.read_csv(csv_path)
    if raw_df.empty:
        return 0

    df = normalize_columns(raw_df)
    required_columns = {"name", "skills"}
    if not required_columns.issubset(df.columns):
        return 0

    imported_count = 0
    uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    source_file = csv_path.name

    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        skills_text = str(row.get("skills", "")).strip()
        email = str(row.get("email", "")).strip()

        if not name or not skills_text:
            continue

        existing = conn.execute(
            """
            SELECT 1
            FROM candidates
            WHERE name = ? AND COALESCE(email, '') = ? AND COALESCE(source_file, '') = ?
            LIMIT 1
            """,
            (name, email, source_file),
        ).fetchone()
        if existing:
            continue

        matched_skills, missing_skills = extract_skills(skills_text)
        score = round(calculate_score(skills_text, job_description) * 100, 2)
        status = "Shortlisted" if score >= shortlist_threshold else "Needs Review"

        conn.execute(
            """
            INSERT INTO candidates (name, score, skills, missing_skills, status, uploaded_at, email, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                score,
                ", ".join(matched_skills),
                ", ".join(missing_skills),
                status,
                uploaded_at,
                email,
                source_file,
            ),
        )
        imported_count += 1

    if imported_count:
        conn.commit()

    return imported_count


def sync_local_datasets(conn, shortlist_threshold=75):
    dataset_files = find_local_dataset_files()
    if not dataset_files:
        return []

    job_description = load_job_description()
    imported_files = []
    for csv_path in dataset_files:
        imported_count = import_candidates_from_csv(
            conn,
            csv_path,
            job_description,
            shortlist_threshold=shortlist_threshold,
        )
        if imported_count:
            imported_files.append((csv_path.name, imported_count))
    return imported_files


def get_candidate_data(conn):
    return pd.read_sql_query(
        """
        SELECT rowid, name, score, skills, missing_skills, status, uploaded_at, email, source_file
        FROM candidates
        ORDER BY score DESC, uploaded_at DESC
        """,
        conn,
    )


def render_metric_card(label, value, caption):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(eyebrow, title, copy):
    st.markdown(
        f"""
        <div class="section-heading">
            <div>
                <div class="section-eyebrow">{eyebrow}</div>
                <div class="section-title">{title}</div>
                <div class="section-copy">{copy}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_skill_tags(skills):
    if not skills:
        st.caption("No matched skills detected yet.")
        return
    tags = "".join([f"<span class='tag'>{skill}</span>" for skill in skills])
    st.markdown(tags, unsafe_allow_html=True)


def format_status_label(status):
    if not status or pd.isna(status):
        return "Pending Review"
    return str(status)


def get_status_badge(status):
    label = format_status_label(status)
    css_class = label.lower().replace(" ", "-")
    return (
        f"<span class='status-pill {css_class}'>"
        f"<span class='status-dot'></span>{label}</span>"
    )


def render_status_legend():
    st.markdown(
        """
        <div class="legend-grid">
            <div class="legend-card">
                <span class="status-pill shortlisted"><span class="status-dot"></span>Shortlisted</span>
                <div class="legend-title">Ready for next round</div>
                <div class="legend-copy">
                    This profile crossed the shortlist threshold and looks strong enough for follow-up.
                </div>
            </div>
            <div class="legend-card">
                <span class="status-pill needs-review"><span class="status-dot"></span>Needs Review</span>
                <div class="legend-title">Analyzed but below target</div>
                <div class="legend-copy">
                    The candidate was scored successfully, but the current match is below your shortlist benchmark.
                </div>
            </div>
            <div class="legend-card">
                <span class="status-pill pending-review"><span class="status-dot"></span>Pending Review</span>
                <div class="legend-title">Status not assigned yet</div>
                <div class="legend-copy">
                    This record exists in storage, but no final review state has been saved for it yet.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_candidate_table(df):
    styled_df = df.copy()
    if "Status" in styled_df.columns:
        styled_df["Status"] = styled_df["Status"].apply(format_status_label)

    def status_styles(value):
        status = format_status_label(value)
        palette = {
            "Shortlisted": "background-color: rgba(20, 184, 166, 0.15); color: #99f6e4; font-weight: 700;",
            "Needs Review": "background-color: rgba(245, 158, 11, 0.15); color: #fde68a; font-weight: 700;",
            "Pending Review": "background-color: rgba(148, 163, 184, 0.14); color: #e2e8f0; font-weight: 700;",
        }
        return palette.get(status, "")

    styler = styled_df.style
    if "Match Score" in styled_df.columns:
        styler = styler.background_gradient(
            subset=["Match Score"],
            cmap="YlGn",
            vmin=0,
            vmax=100,
        )
    if "Status" in styled_df.columns:
        styler = styler.map(status_styles, subset=["Status"])

    return styler.format({"Match Score": "{:.2f}%"}, na_rep="-")


def split_skills(skills_text, limit=None):
    if not skills_text or pd.isna(skills_text):
        return []
    skills = [skill.strip() for skill in str(skills_text).split(",") if skill.strip()]
    return skills[:limit] if limit else skills


def render_candidate_cards(df, title, copy, highlight_status=None):
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    render_section_heading("Spotlight", title, copy)

    if df.empty:
        st.info("No candidates match this view yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    cards_html = []
    for _, row in df.iterrows():
        highlight_class = " highlight" if format_status_label(row.get("status")) == highlight_status else ""
        matched_skills = "".join(
            f"<span class='tag'>{skill}</span>" for skill in split_skills(row.get("skills"), limit=4)
        ) or "<span class='tag subtle'>No matched skills</span>"
        missing_skills = "".join(
            f"<span class='tag subtle'>{skill}</span>" for skill in split_skills(row.get("missing_skills"), limit=4)
        ) or "<span class='tag subtle'>No gaps detected</span>"
        source_file = row.get("source_file") if "source_file" in row else ""

        cards_html.append(
            f"""
            <div class="candidate-card{highlight_class}">
                <div class="candidate-topline">
                    <div>
                        <div class="candidate-name">{row["name"]}</div>
                        <div class="candidate-meta">{row.get("uploaded_at") or "No upload date"} | {source_file or "Manual upload"}</div>
                    </div>
                    <div class="candidate-score">
                        <div class="candidate-score-value">{row["score"]:.2f}%</div>
                        <div class="candidate-score-label">Match Score</div>
                    </div>
                </div>
                <div>{get_status_badge(row.get("status"))}</div>
                <div class="mini-label">Matched Skills</div>
                <div class="tag-wrap">{matched_skills}</div>
                <div class="mini-label">Missing Skills</div>
                <div class="tag-wrap">{missing_skills}</div>
            </div>
            """
        )

    st.markdown(f"<div class='candidate-grid'>{''.join(cards_html)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def login_screen():
    st.markdown(
        """
        <div class="login-wrap">
            <div class="hero-title" style="font-size:2.2rem;">AI Resume Screening</div>
            <div class="hero-subtitle" style="margin-bottom:1.4rem;">
                Review resumes faster with a cleaner dashboard, candidate ranking,
                skill insights, and shortlist-ready results.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    setup_tab, login_tab = st.tabs(["Create Account", "Login"])

    with setup_tab:
        st.markdown(
            """
            <div class="login-note">
                Create your own username and password here. They will be saved locally in this project.
            </div>
            """,
            unsafe_allow_html=True,
        )
        new_username = st.text_input("Create Username", key="create_username")
        new_password = st.text_input("Create Password", type="password", key="create_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")

        if st.button("Save Account", use_container_width=True):
            if not new_username.strip() or not new_password:
                st.error("Enter both username and password.")
            elif len(new_password) < 4:
                st.error("Password should be at least 4 characters.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                save_credentials(new_username, new_password)
                st.success("Account created successfully. You can now login.")

        if credentials_exist():
            saved_credentials = load_credentials()
            st.caption(f"Current saved username: {saved_credentials['username']}")

    with login_tab:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True):
            if not credentials_exist():
                st.error("Create an account first, then login.")
            elif verify_login(username, password):
                st.session_state["login"] = True
                st.session_state["active_user"] = username.strip()
                st.rerun()
            else:
                st.error("Invalid username or password.")


def send_email(email, name, score):
    sender = "yourgmail@gmail.com"
    password = "your_app_password"
    yag = yagmail.SMTP(sender, password)
    subject = "Resume Screening Result"
    body = f"""
    Hello {name},

    Your resume match score is {score}%.

    Thank you for applying.
    HR Team
    """
    yag.send(to=email, subject=subject, contents=body)


def render_dashboard(df):
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">AI Resume Screening & Job Matching System</div>
            <div class="hero-subtitle">
                Track applicant quality, shortlist top talent, and review hiring insights in one place.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        total_candidates = 0
        highest_score = 0
        avg_score = 0
        shortlisted = 0
    else:
        total_candidates = len(df)
        highest_score = round(df["score"].max(), 2)
        avg_score = round(df["score"].mean(), 2)
        shortlisted = int((df["status"] == "Shortlisted").sum())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Candidates", total_candidates, "Applicants in database")
    with col2:
        render_metric_card("Highest Score", f"{highest_score}%", "Best job match found")
    with col3:
        render_metric_card("Average Score", f"{avg_score}%", "Overall screening quality")
    with col4:
        render_metric_card("Shortlisted", shortlisted, "Ready for next round")

    render_section_heading(
        "Decision Flow",
        "Status legend for your screening pipeline",
        "Every candidate moves through one of these states so the dashboard stays easy to read during review.",
    )
    render_status_legend()

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        render_section_heading(
            "Performance",
            "Candidate score overview",
            "Compare the strongest profiles quickly and surface the current best match.",
        )
        if df.empty:
            st.info("Upload resumes to see scoring trends and recommendations.")
        else:
            chart_df = df[["name", "score"]].head(10).set_index("name")
            st.bar_chart(chart_df)

            top_row = df.iloc[0]
            st.markdown(
                f"""
                <div class="info-banner">
                    <strong>Top Candidate:</strong> {top_row["name"]} with a score of
                    <strong>{top_row["score"]}%</strong> and current status
                    {get_status_badge(top_row["status"])}.
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        render_section_heading(
            "Insights",
            "Hiring signals",
            "Track common outcomes and which skills appear most often across imported candidates.",
        )
        if df.empty:
            st.write("No insights available yet.")
        else:
            status_counts = df["status"].fillna("Pending Review").value_counts()
            st.dataframe(
                status_counts.rename_axis("Status").reset_index(name="Count"),
                use_container_width=True,
                hide_index=True,
            )
            top_skills = (
                df["skills"]
                .fillna("")
                .str.split(", ")
                .explode()
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .head(6)
            )
            if not top_skills.empty:
                st.caption("Most common matching skills")
                for skill, count in top_skills.items():
                    st.progress(min(count / max(top_skills.max(), 1), 1.0), text=f"{skill} ({count})")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    render_section_heading(
        "Recent Activity",
        "Latest candidate records",
        "A quick look at recently stored profiles and the decisions currently attached to them.",
    )
    if df.empty:
        st.warning("No candidate data available yet.")
    else:
        preview = df[["name", "score", "skills", "status", "uploaded_at"]].head(8)
        preview = preview.rename(
            columns={
                "name": "Candidate Name",
                "score": "Match Score",
                "skills": "Matched Skills",
                "status": "Status",
                "uploaded_at": "Uploaded At",
            }
        )
        st.dataframe(
            style_candidate_table(preview),
            use_container_width=True,
            hide_index=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if not df.empty:
        render_candidate_cards(
            df[df["status"].fillna("Pending Review") == "Shortlisted"].head(4),
            "Shortlisted candidates",
            "These profiles cleared the current threshold and are the easiest group to review first.",
            highlight_status="Shortlisted",
        )


def render_upload_page(conn):
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    render_section_heading(
        "Role Setup",
        "Job Description Studio",
        "Tune the role requirements here so every uploaded resume is scored against the latest hiring criteria.",
    )
    current_job_description = load_job_description()
    updated_job_description = st.text_area(
        "Job Description",
        value=current_job_description,
        height=220,
        help="This text is used to calculate the resume match score.",
    )

    settings_col1, settings_col2 = st.columns([1, 1])
    with settings_col1:
        shortlist_threshold = st.slider(
            "Shortlist threshold (%)",
            min_value=50,
            max_value=95,
            value=75,
            step=5,
        )
    with settings_col2:
        if st.button("Save Job Description", use_container_width=True):
            save_job_description(updated_job_description)
            st.success("Job description updated successfully.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    render_section_heading(
        "Intake",
        "Resume upload workspace",
        "Drop PDF resumes here to score them instantly and push the strongest profiles into your screening dashboard.",
    )
    uploaded_files = st.file_uploader(
        "Upload one or more resumes",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Analyze Resumes", use_container_width=True):
        results = []
        with st.spinner("Analyzing resumes and generating candidate insights..."):
            for file in uploaded_files:
                resume_text = extract_text(file)
                score = calculate_score(resume_text, updated_job_description)
                matched_skills, missing_skills = extract_skills(resume_text)
                match_score = round(score * 100, 2)
                status = "Shortlisted" if match_score >= shortlist_threshold else "Needs Review"
                uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M")

                candidate_record = {
                    "Candidate Name": file.name,
                    "Match Score": match_score,
                    "Status": status,
                    "Matched Skills": ", ".join(matched_skills),
                    "Missing Skills": ", ".join(missing_skills[:5]),
                }
                results.append(candidate_record)

                conn.execute(
                    """
                    INSERT INTO candidates (name, score, skills, missing_skills, status, uploaded_at, email)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file.name,
                        match_score,
                        ", ".join(matched_skills),
                        ", ".join(missing_skills),
                        status,
                        uploaded_at,
                        "",
                    ),
                )

        conn.commit()
        st.session_state["latest_results"] = pd.DataFrame(results).sort_values(
            by="Match Score", ascending=False
        )

    latest_results = st.session_state.get("latest_results")

    if latest_results is not None and not latest_results.empty:
        st.markdown(
            """
            <div class="info-banner">
                Screening complete. Review the rankings below, export the results, and shortlist the strongest candidates.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(latest_results, use_container_width=True, hide_index=True)

        top_candidate = latest_results.iloc[0]
        col1, col2 = st.columns([1, 1])
        with col1:
            st.success(
                f"Top candidate: {top_candidate['Candidate Name']} ({top_candidate['Match Score']}%)"
            )
            render_skill_tags(
                [skill.strip() for skill in top_candidate["Matched Skills"].split(",") if skill.strip()]
            )
        with col2:
            csv_data = latest_results.to_csv(index=False)
            st.download_button(
                label="Download Ranking CSV",
                data=csv_data,
                file_name="resume_ranking.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.subheader("Score Comparison")
        st.bar_chart(latest_results.set_index("Candidate Name")["Match Score"])
    else:
        st.info("Upload resumes and click Analyze Resumes to generate rankings.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    render_section_heading(
        "Bulk Import",
        "Local dataset import",
        "Bring project CSV files into the app database so they appear alongside uploaded resumes in the dashboard.",
    )
    dataset_files = find_local_dataset_files()
    if not dataset_files:
        st.info("No local dataset CSV file was found in the project folder.")
    else:
        st.caption("CSV files found in this project folder can be imported into the dashboard database.")
        st.write(", ".join(file.name for file in dataset_files))
        if st.button("Import Local CSV Dataset", use_container_width=True):
            imported_files = sync_local_datasets(conn)
            if imported_files:
                imported_summary = ", ".join(
                    f"{file_name}: {count} candidates" for file_name, count in imported_files
                )
                st.success(f"Imported {imported_summary}.")
                st.rerun()
            st.info("This dataset was already imported, or it does not contain usable rows.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_results_page(conn):
    df = get_candidate_data(conn)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    render_section_heading(
        "Explorer",
        "Candidate results explorer",
        "Filter by name, status, and score to review the full screening pipeline in one place.",
    )
    if df.empty:
        st.warning("No results available yet. Upload resumes to populate this page.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    filter_col1, filter_col2, filter_col3 = st.columns([1.2, 1, 1])
    with filter_col1:
        search_term = st.text_input("Search candidate name", placeholder="Type a file name...")
    with filter_col2:
        status_options = ["All"] + sorted(df["status"].fillna("Pending Review").unique().tolist())
        selected_status = st.selectbox("Filter by status", status_options)
    with filter_col3:
        minimum_score = st.slider("Minimum score", 0, 100, 60)

    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[
            filtered_df["name"].str.contains(search_term, case=False, na=False)
        ]
    if selected_status != "All":
        filtered_df = filtered_df[
            filtered_df["status"].fillna("Pending Review") == selected_status
        ]
    filtered_df = filtered_df[filtered_df["score"] >= minimum_score]

    display_df = filtered_df[
        ["name", "score", "skills", "missing_skills", "status", "uploaded_at"]
    ].rename(
        columns={
            "name": "Candidate Name",
            "score": "Match Score",
            "skills": "Matched Skills",
            "missing_skills": "Missing Skills",
            "status": "Status",
            "uploaded_at": "Uploaded At",
        }
    )

    st.markdown(
        """
        <div class="glass-callout">
            Review tip: <strong>Needs Review</strong> means the profile was scored but fell below the current shortlist threshold.
            <strong>Pending Review</strong> means the record exists, but no status was saved yet.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        style_candidate_table(display_df),
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(filtered_df.set_index("name")["score"])

    render_candidate_cards(
        filtered_df.head(6),
        "Candidate detail cards",
        "This card view makes it easier to compare score, status, matched skills, and missing skills without scanning across the table.",
        highlight_status="Shortlisted",
    )

    export_csv = display_df.to_csv(index=False)
    st.download_button(
        "Download Filtered Results",
        data=export_csv,
        file_name="filtered_candidate_results.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


apply_custom_css()
conn = init_db()
auto_imported_files = sync_local_datasets(conn)

if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    login_screen()
    st.stop()

with st.sidebar:
    st.markdown("## Talent Console")
    st.caption("Beautiful resume screening for smarter hiring.")
    choice = option_menu(
        "Navigation",
        ["Dashboard", "Upload Resumes", "Results"],
        icons=["speedometer2", "cloud-upload", "bar-chart"],
        menu_icon="briefcase",
        default_index=0,
        styles={
            "container": {"padding": "0.4rem", "background-color": "transparent"},
            "icon": {"color": "#99f6e4", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "6px 0",
                "padding": "12px",
                "border-radius": "12px",
                "color": "#e2e8f0",
                "--hover-color": "rgba(20, 184, 166, 0.12)",
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg, #14b8a6, #0ea5e9)",
                "color": "white",
                "font-weight": "700",
            },
        },
    )

    sidebar_df = get_candidate_data(conn)
    if not sidebar_df.empty:
        st.markdown("---")
        st.metric("Profiles Stored", len(sidebar_df))
        st.metric("Best Match", f"{round(sidebar_df['score'].max(), 2)}%")

    if auto_imported_files:
        st.markdown("---")
        st.caption(
            "Imported local dataset: "
            + ", ".join(f"{file_name} ({count})" for file_name, count in auto_imported_files)
        )

    st.markdown("---")
    if st.button("Logout", use_container_width=True):
        st.session_state["login"] = False
        st.rerun()


candidate_df = get_candidate_data(conn)

if choice == "Dashboard":
    render_dashboard(candidate_df)
elif choice == "Upload Resumes":
    render_upload_page(conn)
else:
    render_results_page(conn)
