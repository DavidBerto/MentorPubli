import sqlite3
import json
import hashlib
import secrets
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
import threading
from dataclasses import dataclass, asdict
import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ===============================
# MODELO DE USUÁRIO
# ===============================

@dataclass
class User:
    """Modelo de usuário"""
    id: str
    username: str
    email: str
    password_hash: str
    created_at: str
    last_login: str
    is_active: bool = True
    reset_token: Optional[str] = None
    reset_token_expires: Optional[str] = None
    profile_data: Dict = None

# ===============================
# SISTEMA DE AUTENTICAÇÃO
# ===============================

class AuthenticationSystem:
    """Sistema de autenticação e gerenciamento de usuários"""
    
    def __init__(self, db_manager: 'UserDatabaseManager'):
        self.db = db_manager
        self.pepper = "StudyAI_S3cr3t_P3pp3r"  # Em produção, usar variável de ambiente
    
    def _hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """Hash de senha com salt e pepper"""
        if not salt:
            salt = secrets.token_hex(32)
        
        # Combinar senha + salt + pepper
        combined = f"{password}{salt}{self.pepper}"
        
        # Hash com SHA-256
        password_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return password_hash, salt
    
    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verifica se a senha está correta"""
        test_hash, _ = self._hash_password(password, salt)
        return test_hash == password_hash
    
    def validate_password(self, password: str) -> Tuple[bool, str]:
        """Valida força da senha"""
        if len(password) < 8:
            return False, "Senha deve ter no mínimo 8 caracteres"
        
        if not re.search(r'[A-Z]', password):
            return False, "Senha deve conter pelo menos uma letra maiúscula"
        
        if not re.search(r'[a-z]', password):
            return False, "Senha deve conter pelo menos uma letra minúscula"
        
        if not re.search(r'\d', password):
            return False, "Senha deve conter pelo menos um número"
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Senha deve conter pelo menos um caractere especial"
        
        return True, "Senha válida"
    
    def validate_email(self, email: str) -> bool:
        """Valida formato do email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def register_user(self, username: str, email: str, password: str) -> Tuple[bool, str]:
        """Registra novo usuário"""
        # Validações
        if not username or len(username) < 3:
            return False, "Nome de usuário deve ter no mínimo 3 caracteres"
        
        if not self.validate_email(email):
            return False, "Email inválido"
        
        valid, msg = self.validate_password(password)
        if not valid:
            return False, msg
        
        # Verificar se usuário já existe
        if self.db.get_user_by_username(username):
            return False, "Nome de usuário já existe"
        
        if self.db.get_user_by_email(email):
            return False, "Email já cadastrado"
        
        # Criar hash da senha
        password_hash, salt = self._hash_password(password)
        
        # Criar usuário
        user_id = hashlib.md5(f"{username}{datetime.now().isoformat()}".encode()).hexdigest()
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=f"{password_hash}:{salt}",  # Armazenar hash:salt
            created_at=datetime.now().isoformat(),
            last_login="",
            is_active=True,
            profile_data={}
        )
        
        # Salvar no banco
        self.db.create_user(user)
        
        return True, "Usuário criado com sucesso!"
    
    def login(self, username_or_email: str, password: str) -> Tuple[bool, Optional[User], str]:
        """Realiza login do usuário"""
        # Buscar usuário por username ou email
        user = self.db.get_user_by_username(username_or_email)
        if not user:
            user = self.db.get_user_by_email(username_or_email)
        
        if not user:
            return False, None, "Usuário não encontrado"
        
        if not user.is_active:
            return False, None, "Conta desativada"
        
        # Verificar senha
        hash_parts = user.password_hash.split(':')
        if len(hash_parts) != 2:
            return False, None, "Erro na verificação da senha"
        
        stored_hash, salt = hash_parts
        
        if self._verify_password(password, stored_hash, salt):
            # Atualizar último login
            user.last_login = datetime.now().isoformat()
            self.db.update_user(user)
            
            return True, user, "Login realizado com sucesso!"
        
        return False, None, "Senha incorreta"
    
    def generate_reset_token(self, email: str) -> Tuple[bool, str, str]:
        """Gera token para recuperação de senha"""
        user = self.db.get_user_by_email(email)
        
        if not user:
            return False, "", "Email não encontrado"
        
        # Gerar token único
        token = secrets.token_urlsafe(32)
        expires = (datetime.now() + timedelta(hours=1)).isoformat()
        
        # Salvar token no usuário
        user.reset_token = token
        user.reset_token_expires = expires
        self.db.update_user(user)
        
        return True, token, "Token gerado com sucesso"
    
    def reset_password(self, token: str, new_password: str) -> Tuple[bool, str]:
        """Reseta senha usando token"""
        # Buscar usuário pelo token
        user = self.db.get_user_by_reset_token(token)
        
        if not user:
            return False, "Token inválido"
        
        # Verificar expiração
        if user.reset_token_expires:
            expires = datetime.fromisoformat(user.reset_token_expires)
            if datetime.now() > expires:
                return False, "Token expirado"
        
        # Validar nova senha
        valid, msg = self.validate_password(new_password)
        if not valid:
            return False, msg
        
        # Atualizar senha
        password_hash, salt = self._hash_password(new_password)
        user.password_hash = f"{password_hash}:{salt}"
        user.reset_token = None
        user.reset_token_expires = None
        
        self.db.update_user(user)
        
        return True, "Senha alterada com sucesso!"
    
    def change_password(self, user_id: str, current_password: str, new_password: str) -> Tuple[bool, str]:
        """Altera senha do usuário autenticado"""
        user = self.db.get_user_by_id(user_id)
        
        if not user:
            return False, "Usuário não encontrado"
        
        # Verificar senha atual
        hash_parts = user.password_hash.split(':')
        if len(hash_parts) != 2:
            return False, "Erro na verificação da senha"
        
        stored_hash, salt = hash_parts
        
        if not self._verify_password(current_password, stored_hash, salt):
            return False, "Senha atual incorreta"
        
        # Validar nova senha
        valid, msg = self.validate_password(new_password)
        if not valid:
            return False, msg
        
        # Atualizar senha
        password_hash, salt = self._hash_password(new_password)
        user.password_hash = f"{password_hash}:{salt}"
        
        self.db.update_user(user)
        
        return True, "Senha alterada com sucesso!"

# ===============================
# BANCO DE DADOS COM USUÁRIOS
# ===============================

class UserDatabaseManager:
    """Banco de dados com suporte a múltiplos usuários"""
    
    def __init__(self, db_path: str = "studyai_multiuser.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._initialize_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager para conexões thread-safe"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
    
    def _initialize_database(self):
        """Cria as tabelas com suporte a usuários"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tabela de usuários
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    reset_token TEXT,
                    reset_token_expires TIMESTAMP,
                    profile_data TEXT
                )
            """)
            
            # Atualizar tabelas existentes para incluir user_id
            
            # Tabela de notas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT,
                    connections TEXT,
                    category TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Tabela de flashcards
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS flashcards (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    difficulty INTEGER,
                    category TEXT,
                    last_reviewed TIMESTAMP,
                    next_review TIMESTAMP,
                    correct_count INTEGER DEFAULT 0,
                    total_reviews INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Tabela de quizzes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quizzes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    questions TEXT NOT NULL,
                    category TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Tabela de sessões de estudo
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS study_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    activity TEXT NOT NULL,
                    duration INTEGER,
                    category TEXT,
                    score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Configurações por usuário
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_config (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, key),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Índices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_notes_user ON notes(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_user ON flashcards(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quizzes_user ON quizzes(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON study_sessions(user_id)")
    
    # ===============================
    # MÉTODOS DE USUÁRIO
    # ===============================
    
    def create_user(self, user: User):
        """Cria novo usuário"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users 
                (id, username, email, password_hash, created_at, last_login, 
                 is_active, reset_token, reset_token_expires, profile_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user.id, user.username, user.email, user.password_hash,
                user.created_at, user.last_login, user.is_active,
                user.reset_token, user.reset_token_expires,
                json.dumps(user.profile_data) if user.profile_data else '{}'
            ))
    
    def update_user(self, user: User):
        """Atualiza dados do usuário"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET
                    email = ?, password_hash = ?, last_login = ?,
                    is_active = ?, reset_token = ?, reset_token_expires = ?,
                    profile_data = ?
                WHERE id = ?
            """, (
                user.email, user.password_hash, user.last_login,
                user.is_active, user.reset_token, user.reset_token_expires,
                json.dumps(user.profile_data) if user.profile_data else '{}',
                user.id
            ))
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Busca usuário por ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            row = cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            
            if row:
                return User(
                    id=row['id'],
                    username=row['username'],
                    email=row['email'],
                    password_hash=row['password_hash'],
                    created_at=row['created_at'],
                    last_login=row['last_login'] or "",
                    is_active=bool(row['is_active']),
                    reset_token=row['reset_token'],
                    reset_token_expires=row['reset_token_expires'],
                    profile_data=json.loads(row['profile_data']) if row['profile_data'] else {}
                )
        return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Busca usuário por username"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            row = cursor.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            
            if row:
                return self._row_to_user(row)
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Busca usuário por email"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            row = cursor.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            
            if row:
                return self._row_to_user(row)
        return None
    
    def get_user_by_reset_token(self, token: str) -> Optional[User]:
        """Busca usuário por token de reset"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            row = cursor.execute("SELECT * FROM users WHERE reset_token = ?", (token,)).fetchone()
            
            if row:
                return self._row_to_user(row)
        return None
    
    def _row_to_user(self, row) -> User:
        """Converte row do banco em objeto User"""
        return User(
            id=row['id'],
            username=row['username'],
            email=row['email'],
            password_hash=row['password_hash'],
            created_at=row['created_at'],
            last_login=row['last_login'] or "",
            is_active=bool(row['is_active']),
            reset_token=row['reset_token'],
            reset_token_expires=row['reset_token_expires'],
            profile_data=json.loads(row['profile_data']) if row['profile_data'] else {}
        )
    
    # ===============================
    # MÉTODOS COM USER_ID
    # ===============================
    
    def save_note(self, note: 'Note', user_id: str):
        """Salva nota para usuário específico"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO notes 
                (id, user_id, title, content, tags, connections, category, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                note.id, user_id, note.title, note.content,
                json.dumps(note.tags), json.dumps(note.connections),
                note.category, note.created_at
            ))
    
    def get_user_notes(self, user_id: str) -> Dict:
        """Recupera notas do usuário"""
        notes = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            rows = cursor.execute(
                "SELECT * FROM notes WHERE user_id = ? ORDER BY created_at DESC", 
                (user_id,)
            ).fetchall()
            
            for row in rows:
                # Importar classe Note localmente
                from dataclasses import dataclass
                @dataclass
                class Note:
                    id: str
                    title: str
                    content: str
                    tags: List[str]
                    connections: List[str]
                    created_at: str
                    category: str
                
                note = Note(
                    id=row['id'],
                    title=row['title'],
                    content=row['content'],
                    tags=json.loads(row['tags']),
                    connections=json.loads(row['connections']),
                    created_at=row['created_at'],
                    category=row['category']
                )
                notes[note.id] = note
        
        return notes
    
    def save_flashcard(self, flashcard: 'Flashcard', user_id: str):
        """Salva flashcard para usuário específico"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO flashcards 
                (id, user_id, question, answer, difficulty, category, 
                 last_reviewed, next_review, correct_count, total_reviews)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                flashcard.id, user_id, flashcard.question, flashcard.answer,
                flashcard.difficulty, flashcard.category, flashcard.last_reviewed,
                flashcard.next_review, flashcard.correct_count, flashcard.total_reviews
            ))
    
    def get_user_flashcards(self, user_id: str) -> Dict:
        """Recupera flashcards do usuário"""
        flashcards = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            rows = cursor.execute(
                "SELECT * FROM flashcards WHERE user_id = ?", 
                (user_id,)
            ).fetchall()
            
            for row in rows:
                from dataclasses import dataclass
                @dataclass
                class Flashcard:
                    id: str
                    question: str
                    answer: str
                    difficulty: int
                    category: str
                    last_reviewed: str
                    next_review: str
                    correct_count: int
                    total_reviews: int
                    
                    @property
                    def accuracy(self) -> float:
                        if self.total_reviews == 0:
                            return 0.0
                        return (self.correct_count / self.total_reviews) * 100
                
                flashcard = Flashcard(
                    id=row['id'],
                    question=row['question'],
                    answer=row['answer'],
                    difficulty=row['difficulty'],
                    category=row['category'],
                    last_reviewed=row['last_reviewed'] or "",
                    next_review=row['next_review'] or "",
                    correct_count=row['correct_count'],
                    total_reviews=row['total_reviews']
                )
                flashcards[flashcard.id] = flashcard
        
        return flashcards
    
    def save_quiz(self, quiz: 'Quiz', user_id: str):
        """Salva quiz para usuário específico"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO quizzes 
                (id, user_id, title, questions, category, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                quiz.id, user_id, quiz.title,
                json.dumps(quiz.questions), quiz.category, quiz.created_at
            ))
    
    def get_user_quizzes(self, user_id: str) -> Dict:
        """Recupera quizzes do usuário"""
        quizzes = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            rows = cursor.execute(
                "SELECT * FROM quizzes WHERE user_id = ? ORDER BY created_at DESC", 
                (user_id,)
            ).fetchall()
            
            for row in rows:
                from dataclasses import dataclass
                @dataclass
                class Quiz:
                    id: str
                    title: str
                    questions: List[Dict]
                    category: str
                    created_at: str
                
                quiz = Quiz(
                    id=row['id'],
                    title=row['title'],
                    questions=json.loads(row['questions']),
                    category=row['category'],
                    created_at=row['created_at']
                )
                quizzes[quiz.id] = quiz
        
        return quizzes
    
    def save_study_session(self, session: 'StudySession', user_id: str):
        """Salva sessão de estudo para usuário específico"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO study_sessions 
                (user_id, date, activity, duration, category, score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id, session.date, session.activity,
                session.duration, session.category, session.score
            ))
    
    def get_user_study_sessions(self, user_id: str) -> List:
        """Recupera sessões de estudo do usuário"""
        sessions = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            rows = cursor.execute("""
                SELECT date, activity, duration, category, score 
                FROM study_sessions 
                WHERE user_id = ?
                ORDER BY date DESC
            """, (user_id,)).fetchall()
            
            for row in rows:
                from dataclasses import dataclass
                @dataclass
                class StudySession:
                    date: str
                    activity: str
                    duration: int
                    category: str
                    score: float
                
                session = StudySession(
                    date=row['date'],
                    activity=row['activity'],
                    duration=row['duration'],
                    category=row['category'],
                    score=row['score']
                )
                sessions.append(session)
        
        return sessions
    
    def save_user_config(self, user_id: str, key: str, value: Any):
        """Salva configuração do usuário"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_config (user_id, key, value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (user_id, key, json.dumps(value) if not isinstance(value, str) else value))
    
    def get_user_config(self, user_id: str, key: str, default=None):
        """Recupera configuração do usuário"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            result = cursor.execute(
                "SELECT value FROM user_config WHERE user_id = ? AND key = ?", 
                (user_id, key)
            ).fetchone()
            
            if result:
                try:
                    return json.loads(result['value'])
                except:
                    return result['value']
            return default

# ===============================
# PÁGINAS DE AUTENTICAÇÃO
# ===============================

class AuthPages:
    """Páginas de login, registro e recuperação de senha"""
    
    @staticmethod
    def render_login_page():
        """Página de login"""
        st.markdown("### 🔐 Login")
        
        with st.form("login_form"):
            username_or_email = st.text_input(
                "Usuário ou Email",
                placeholder="Digite seu nome de usuário ou email"
            )
            
            password = st.text_input(
                "Senha",
                type="password",
                placeholder="Digite sua senha"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                login_btn = st.form_submit_button("🚪 Entrar", use_container_width=True)
            
            with col2:
                register_btn = st.form_submit_button("📝 Criar Conta", use_container_width=True)
        
        # Links adicionais
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔑 Esqueci minha senha", use_container_width=True):
                st.session_state.auth_page = "forgot_password"
                st.rerun()
        
        with col2:
            if st.button("📝 Novo usuário", use_container_width=True):
                st.session_state.auth_page = "register"
                st.rerun()
        
        # Processar login
        if login_btn:
            if username_or_email and password:
                auth = AuthenticationSystem(st.session_state.db_manager)
                success, user, message = auth.login(username_or_email, password)
                
                if success:
                    st.session_state.current_user = user
                    st.session_state.logged_in = True
                    st.success(message)
                    st.balloons()
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Preencha todos os campos")
        
        # Ir para registro
        if register_btn:
            st.session_state.auth_page = "register"
            st.rerun()
    
    @staticmethod
    def render_register_page():
        """Página de registro"""
        st.markdown("### 📝 Criar Conta")
        
        with st.form("register_form"):
            username = st.text_input(
                "Nome de usuário",
                placeholder="Escolha um nome de usuário único",
                help="Mínimo 3 caracteres"
            )
            
            email = st.text_input(
                "Email",
                placeholder="seu@email.com"
            )
            
            password = st.text_input(
                "Senha",
                type="password",
                placeholder="Crie uma senha forte",
                help="Mínimo 8 caracteres, com maiúsculas, minúsculas, números e símbolos"
            )
            
            password_confirm = st.text_input(
                "Confirmar senha",
                type="password",
                placeholder="Digite a senha novamente"
            )
            
            terms = st.checkbox("Aceito os termos de uso e política de privacidade")
            
            col1, col2 = st.columns(2)
            
            with col1:
                register_btn = st.form_submit_button("✅ Criar Conta", use_container_width=True)
            
            with col2:
                back_btn = st.form_submit_button("⬅️ Voltar", use_container_width=True)
        
        # Mostrar requisitos de senha
        st.info("""
        **Requisitos da senha:**
        - Mínimo 8 caracteres
        - Pelo menos 1 letra maiúscula
        - Pelo menos 1 letra minúscula
        - Pelo menos 1 número
        - Pelo menos 1 caractere especial (!@#$%^&*...)
        """)
        
        # Processar registro
        if register_btn:
            if not terms:
                st.error("Você deve aceitar os termos de uso")
            elif password != password_confirm:
                st.error("As senhas não coincidem")
            elif username and email and password:
                auth = AuthenticationSystem(st.session_state.db_manager)
                success, message = auth.register_user(username, email, password)
                
                if success:
                    st.success(message)
                    st.info("Faça login para continuar")
                    st.session_state.auth_page = "login"
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Preencha todos os campos")
        
        # Voltar ao login
        if back_btn:
            st.session_state.auth_page = "login"
            st.rerun()
    
    @staticmethod
    def render_forgot_password_page():
        """Página de recuperação de senha"""
        st.markdown("### 🔑 Recuperar Senha")
        
        if 'reset_token_sent' not in st.session_state:
            # Etapa 1: Solicitar email
            with st.form("forgot_password_form"):
                email = st.text_input(
                    "Email cadastrado",
                    placeholder="Digite o email da sua conta"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    send_btn = st.form_submit_button("📧 Enviar Token", use_container_width=True)
                
                with col2:
                    back_btn = st.form_submit_button("⬅️ Voltar", use_container_width=True)
            
            if send_btn and email:
                auth = AuthenticationSystem(st.session_state.db_manager)
                success, token, message = auth.generate_reset_token(email)
                
                if success:
                    # Em produção, enviar por email
                    st.success("Token gerado com sucesso!")
                    st.info(f"Token de recuperação: `{token}`")
                    st.warning("Em produção, este token seria enviado por email")
                    st.session_state.reset_token_sent = True
                    st.session_state.reset_email = email
                    st.rerun()
                else:
                    st.error(message)
            
            if back_btn:
                st.session_state.auth_page = "login"
                st.rerun()
        
        else:
            # Etapa 2: Resetar senha com token
            st.info(f"Token enviado para: {st.session_state.reset_email}")
            
            with st.form("reset_password_form"):
                token = st.text_input(
                    "Token de recuperação",
                    placeholder="Cole o token recebido"
                )
                
                new_password = st.text_input(
                    "Nova senha",
                    type="password",
                    placeholder="Digite sua nova senha"
                )
                
                confirm_password = st.text_input(
                    "Confirmar nova senha",
                    type="password",
                    placeholder="Digite a senha novamente"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    reset_btn = st.form_submit_button("🔄 Resetar Senha", use_container_width=True)
                
                with col2:
                    cancel_btn = st.form_submit_button("❌ Cancelar", use_container_width=True)
            
            if reset_btn:
                if new_password != confirm_password:
                    st.error("As senhas não coincidem")
                elif token and new_password:
                    auth = AuthenticationSystem(st.session_state.db_manager)
                    success, message = auth.reset_password(token, new_password)
                    
                    if success:
                        st.success(message)
                        del st.session_state.reset_token_sent
                        del st.session_state.reset_email
                        st.session_state.auth_page = "login"
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Preencha todos os campos")
            
            if cancel_btn:
                del st.session_state.reset_token_sent
                del st.session_state.reset_email
                st.session_state.auth_page = "login"
                st.rerun()

# ===============================
# PÁGINA DE PERFIL DO USUÁRIO
# ===============================

class UserProfilePage:
    """Página de perfil e configurações do usuário"""
    
    @staticmethod
    def render():
        """Renderiza página de perfil"""
        if 'current_user' not in st.session_state:
            st.error("Você precisa estar logado")
            return
        
        user = st.session_state.current_user
        db = st.session_state.db_manager
        
        st.markdown(f"### 👤 Perfil de {user.username}")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Estatísticas", "⚙️ Configurações", "🔒 Segurança", "📤 Exportar"])
        
        with tab1:
            UserProfilePage._render_statistics(user, db)
        
        with tab2:
            UserProfilePage._render_settings(user, db)
        
        with tab3:
            UserProfilePage._render_security(user, db)
        
        with tab4:
            UserProfilePage._render_export(user, db)
    
    @staticmethod
    def _render_statistics(user: User, db: UserDatabaseManager):
        """Renderiza estatísticas do usuário"""
        # Coletar estatísticas
        notes_count = len(db.get_user_notes(user.id))
        flashcards_count = len(db.get_user_flashcards(user.id))
        quizzes_count = len(db.get_user_quizzes(user.id))
        sessions = db.get_user_study_sessions(user.id)
        
        total_time = sum([s.duration for s in sessions])
        
        # Mostrar métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Notas", notes_count)
        
        with col2:
            st.metric("🗃️ Flashcards", flashcards_count)
        
        with col3:
            st.metric("📊 Quizzes", quizzes_count)
        
        with col4:
            st.metric("⏱️ Tempo Total", f"{total_time} min")
        
        # Informações da conta
        st.markdown("#### 📋 Informações da Conta")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Usuário:** {user.username}")
            st.write(f"**Email:** {user.email}")
        
        with col2:
            created = datetime.fromisoformat(user.created_at).strftime("%d/%m/%Y")
            st.write(f"**Conta criada:** {created}")
            
            if user.last_login:
                last_login = datetime.fromisoformat(user.last_login).strftime("%d/%m/%Y %H:%M")
                st.write(f"**Último login:** {last_login}")
    
    @staticmethod
    def _render_settings(user: User, db: UserDatabaseManager):
        """Renderiza configurações do usuário"""
        st.markdown("#### ⚙️ Configurações da Conta")
        
        # Email
        with st.form("update_email_form"):
            new_email = st.text_input("Email", value=user.email)
            
            if st.form_submit_button("💾 Atualizar Email"):
                if new_email and new_email != user.email:
                    auth = AuthenticationSystem(db)
                    if auth.validate_email(new_email):
                        # Verificar se email já existe
                        if not db.get_user_by_email(new_email):
                            user.email = new_email
                            db.update_user(user)
                            st.success("Email atualizado com sucesso!")
                            st.rerun()
                        else:
                            st.error("Este email já está em uso")
                    else:
                        st.error("Email inválido")
        
        # Tema
        st.markdown("#### 🎨 Preferências")
        
        theme = db.get_user_config(user.id, 'theme', 'light')
        new_theme = st.selectbox("Tema", ['light', 'dark'], index=0 if theme == 'light' else 1)
        
        if new_theme != theme:
            db.save_user_config(user.id, 'theme', new_theme)
            st.success("Tema atualizado!")
        
        # Notificações
        notifications = db.get_user_config(user.id, 'notifications', True)
        new_notifications = st.checkbox("Receber notificações", value=notifications)
        
        if new_notifications != notifications:
            db.save_user_config(user.id, 'notifications', new_notifications)
    
    @staticmethod
    def _render_security(user: User, db: UserDatabaseManager):
        """Renderiza configurações de segurança"""
        st.markdown("#### 🔒 Segurança")
        
        # Alterar senha
        with st.form("change_password_form"):
            current_password = st.text_input("Senha atual", type="password")
            new_password = st.text_input("Nova senha", type="password")
            confirm_password = st.text_input("Confirmar nova senha", type="password")
            
            if st.form_submit_button("🔄 Alterar Senha"):
                if new_password != confirm_password:
                    st.error("As senhas não coincidem")
                elif current_password and new_password:
                    auth = AuthenticationSystem(db)
                    success, message = auth.change_password(
                        user.id, current_password, new_password
                    )
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.error("Preencha todos os campos")
        
        # Logout
        st.markdown("#### 🚪 Sessão")
        if st.button("🚪 Fazer Logout", use_container_width=True):
            del st.session_state.current_user
            st.session_state.logged_in = False
            st.rerun()
    
    @staticmethod
    def _render_export(user: User, db: UserDatabaseManager):
        """Renderiza opções de exportação"""
        st.markdown("#### 📤 Exportar Dados")
        
        if st.button("💾 Exportar meus dados (JSON)", use_container_width=True):
            # Coletar todos os dados do usuário
            export_data = {
                'user_info': {
                    'username': user.username,
                    'email': user.email,
                    'created_at': user.created_at
                },
                'notes': {k: asdict(v) for k, v in db.get_user_notes(user.id).items()},
                'flashcards': {k: asdict(v) for k, v in db.get_user_flashcards(user.id).items()},
                'quizzes': {k: asdict(v) for k, v in db.get_user_quizzes(user.id).items()},
                'sessions': [asdict(s) for s in db.get_user_study_sessions(user.id)]
            }
            
            # Criar download
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
            
            st.download_button(
                label="⬇️ Baixar dados",
                data=json_str,
                file_name=f"studyai_export_{user.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

# ===============================
# SESSION MANAGER ATUALIZADO
# ===============================

class UserAwareSessionManager:
    """SessionManager com suporte a múltiplos usuários"""
    
    @staticmethod
    def init_session_state():
        """Inicializa estado com suporte a usuários"""
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = UserDatabaseManager()
        
        # Estado de autenticação
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        
        if 'auth_page' not in st.session_state:
            st.session_state.auth_page = "login"
        
        # Se logado, carregar dados do usuário
        if st.session_state.logged_in and 'current_user' in st.session_state:
            user = st.session_state.current_user
            db = st.session_state.db_manager
            
            # Carregar dados do usuário
            if 'user_data_loaded' not in st.session_state:
                st.session_state.notes = db.get_user_notes(user.id)
                st.session_state.flashcards = db.get_user_flashcards(user.id)
                st.session_state.quizzes = db.get_user_quizzes(user.id)
                st.session_state.study_sessions = db.get_user_study_sessions(user.id)
                
                # Carregar configurações
                api_key = db.get_user_config(user.id, 'openai_api_key')
                if api_key:
                    st.session_state.openai_api_key = api_key
                
                st.session_state.user_data_loaded = True
        
        # Outros estados
        defaults = {
            'current_page': "Dashboard",
            'latest_insights': None,
            'study_mode': False,
            'current_card': None,
            'show_answer': False,
            'quiz_answers': {},
            'quiz_submitted': False,
            'active_quiz': None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def save_note(note):
        """Salva nota para o usuário atual"""
        if 'current_user' in st.session_state:
            user_id = st.session_state.current_user.id
            st.session_state.notes[note.id] = note
            st.session_state.db_manager.save_note(note, user_id)
    
    @staticmethod
    def save_flashcard(flashcard):
        """Salva flashcard para o usuário atual"""
        if 'current_user' in st.session_state:
            user_id = st.session_state.current_user.id
            st.session_state.flashcards[flashcard.id] = flashcard
            st.session_state.db_manager.save_flashcard(flashcard, user_id)
    
    @staticmethod
    def save_quiz(quiz):
        """Salva quiz para o usuário atual"""
        if 'current_user' in st.session_state:
            user_id = st.session_state.current_user.id
            st.session_state.quizzes[quiz.id] = quiz
            st.session_state.db_manager.save_quiz(quiz, user_id)
    
    @staticmethod
    def save_study_session(session):
        """Salva sessão de estudo para o usuário atual"""
        if 'current_user' in st.session_state:
            user_id = st.session_state.current_user.id
            st.session_state.study_sessions.append(session)
            st.session_state.db_manager.save_study_session(session, user_id)
    
    @staticmethod
    def save_api_key(api_key):
        """Salva chave da API para o usuário atual"""
        if 'current_user' in st.session_state:
            user_id = st.session_state.current_user.id
            st.session_state.openai_api_key = api_key
            st.session_state.db_manager.save_user_config(user_id, 'openai_api_key', api_key)
