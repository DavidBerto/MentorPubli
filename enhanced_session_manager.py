import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import threading
from dataclasses import asdict
import shutil
import streamlit as st

class DatabaseManager:
    """Gerenciador de banco de dados SQLite para persistência local"""
    
    def __init__(self, db_path: str = "studyai_data.db"):
        self.db_path = db_path
        self.backup_path = f"{db_path}.backup"
        self._lock = threading.Lock()
        self._initialize_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager para conexões thread-safe"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Permite acesso por nome de coluna
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
    
    def _initialize_database(self):
        """Cria as tabelas se não existirem"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tabela de configurações
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabela de notas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT,
                    connections TEXT,
                    category TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabela de flashcards
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS flashcards (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    difficulty INTEGER,
                    category TEXT,
                    last_reviewed TIMESTAMP,
                    next_review TIMESTAMP,
                    correct_count INTEGER DEFAULT 0,
                    total_reviews INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabela de quizzes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quizzes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    questions TEXT NOT NULL,
                    category TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabela de sessões de estudo
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS study_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    activity TEXT NOT NULL,
                    duration INTEGER,
                    category TEXT,
                    score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Índices para melhor performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_notes_category ON notes(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_category ON flashcards(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_date ON study_sessions(date)")
    
    # ===============================
    # MÉTODOS PARA CONFIGURAÇÕES
    # ===============================
    
    def save_config(self, key: str, value: Any):
        """Salva uma configuração"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO config (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, json.dumps(value) if not isinstance(value, str) else value))
    
    def get_config(self, key: str, default=None):
        """Recupera uma configuração"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            result = cursor.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
            if result:
                try:
                    return json.loads(result['value'])
                except:
                    return result['value']
            return default
    
    # ===============================
    # MÉTODOS PARA NOTAS
    # ===============================
    
    def save_note(self, note: 'Note'):
        """Salva ou atualiza uma nota"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO notes 
                (id, title, content, tags, connections, category, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                note.id,
                note.title,
                note.content,
                json.dumps(note.tags),
                json.dumps(note.connections),
                note.category,
                note.created_at
            ))
    
    def get_all_notes(self) -> Dict[str, 'Note']:
        """Recupera todas as notas"""
        notes = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            rows = cursor.execute("SELECT * FROM notes ORDER BY created_at DESC").fetchall()
            
            for row in rows:
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
    
    def delete_note(self, note_id: str):
        """Deleta uma nota"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    
    # ===============================
    # MÉTODOS PARA FLASHCARDS
    # ===============================
    
    def save_flashcard(self, flashcard: 'Flashcard'):
        """Salva ou atualiza um flashcard"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO flashcards 
                (id, question, answer, difficulty, category, last_reviewed, 
                 next_review, correct_count, total_reviews)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                flashcard.id,
                flashcard.question,
                flashcard.answer,
                flashcard.difficulty,
                flashcard.category,
                flashcard.last_reviewed,
                flashcard.next_review,
                flashcard.correct_count,
                flashcard.total_reviews
            ))
    
    def get_all_flashcards(self) -> Dict[str, 'Flashcard']:
        """Recupera todos os flashcards"""
        flashcards = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            rows = cursor.execute("SELECT * FROM flashcards").fetchall()
            
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
    
    def delete_flashcard(self, flashcard_id: str):
        """Deleta um flashcard"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM flashcards WHERE id = ?", (flashcard_id,))
    
    # ===============================
    # MÉTODOS PARA QUIZZES
    # ===============================
    
    def save_quiz(self, quiz: 'Quiz'):
        """Salva ou atualiza um quiz"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO quizzes 
                (id, title, questions, category, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                quiz.id,
                quiz.title,
                json.dumps(quiz.questions),
                quiz.category,
                quiz.created_at
            ))
    
    def get_all_quizzes(self) -> Dict[str, 'Quiz']:
        """Recupera todos os quizzes"""
        quizzes = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            rows = cursor.execute("SELECT * FROM quizzes ORDER BY created_at DESC").fetchall()
            
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
    
    def delete_quiz(self, quiz_id: str):
        """Deleta um quiz"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM quizzes WHERE id = ?", (quiz_id,))
    
    # ===============================
    # MÉTODOS PARA SESSÕES DE ESTUDO
    # ===============================
    
    def save_study_session(self, session: 'StudySession'):
        """Salva uma sessão de estudo"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO study_sessions 
                (date, activity, duration, category, score)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session.date,
                session.activity,
                session.duration,
                session.category,
                session.score
            ))
    
    def get_all_study_sessions(self) -> List['StudySession']:
        """Recupera todas as sessões de estudo"""
        sessions = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            rows = cursor.execute("""
                SELECT date, activity, duration, category, score 
                FROM study_sessions 
                ORDER BY date DESC
            """).fetchall()
            
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
    
    # ===============================
    # MÉTODOS DE BACKUP E RESTAURAÇÃO
    # ===============================
    
    def create_backup(self):
        """Cria backup do banco de dados"""
        try:
            shutil.copy2(self.db_path, self.backup_path)
            return True
        except Exception as e:
            print(f"Erro ao criar backup: {e}")
            return False
    
    def restore_backup(self):
        """Restaura banco de dados do backup"""
        if os.path.exists(self.backup_path):
            try:
                shutil.copy2(self.backup_path, self.db_path)
                return True
            except Exception as e:
                print(f"Erro ao restaurar backup: {e}")
                return False
        return False
    
    def export_to_json(self, filepath: str):
        """Exporta todos os dados para JSON"""
        data = {
            'export_date': datetime.now().isoformat(),
            'notes': {k: asdict(v) for k, v in self.get_all_notes().items()},
            'flashcards': {k: asdict(v) for k, v in self.get_all_flashcards().items()},
            'quizzes': {k: asdict(v) for k, v in self.get_all_quizzes().items()},
            'sessions': [asdict(s) for s in self.get_all_study_sessions()],
            'config': self._get_all_config()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def import_from_json(self, filepath: str):
        """Importa dados de arquivo JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Importar notas
        for note_data in data.get('notes', {}).values():
            note = type('Note', (), note_data)()
            self.save_note(note)
        
        # Importar flashcards
        for card_data in data.get('flashcards', {}).values():
            card = type('Flashcard', (), card_data)()
            self.save_flashcard(card)
        
        # Importar quizzes
        for quiz_data in data.get('quizzes', {}).values():
            quiz = type('Quiz', (), quiz_data)()
            self.save_quiz(quiz)
        
        # Importar sessões
        for session_data in data.get('sessions', []):
            session = type('StudySession', (), session_data)()
            self.save_study_session(session)
    
    def _get_all_config(self) -> Dict[str, Any]:
        """Recupera todas as configurações"""
        config = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            rows = cursor.execute("SELECT key, value FROM config").fetchall()
            for row in rows:
                try:
                    config[row['key']] = json.loads(row['value'])
                except:
                    config[row['key']] = row['value']
        return config
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do banco de dados"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {
                'total_notes': cursor.execute("SELECT COUNT(*) FROM notes").fetchone()[0],
                'total_flashcards': cursor.execute("SELECT COUNT(*) FROM flashcards").fetchone()[0],
                'total_quizzes': cursor.execute("SELECT COUNT(*) FROM quizzes").fetchone()[0],
                'total_sessions': cursor.execute("SELECT COUNT(*) FROM study_sessions").fetchone()[0],
                'total_study_time': cursor.execute("SELECT SUM(duration) FROM study_sessions").fetchone()[0] or 0,
                'database_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            }
            
            return stats


# ===============================
# INTEGRAÇÃO COM SESSION MANAGER
# ===============================

class EnhancedSessionManager:
    """SessionManager melhorado com persistência SQLite"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    @staticmethod
    def init_session_state():
        """Inicializa estado e carrega dados do banco"""
        # Inicializar banco de dados
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = DatabaseManager()
        
        db = st.session_state.db_manager
        
        # Carregar dados do banco se ainda não carregados
        if 'data_loaded' not in st.session_state:
            try:
                # Carregar configurações
                api_key = db.get_config('openai_api_key')
                if api_key:
                    st.session_state.openai_api_key = api_key
                
                # Carregar dados
                st.session_state.notes = db.get_all_notes()
                st.session_state.flashcards = db.get_all_flashcards()
                st.session_state.quizzes = db.get_all_quizzes()
                st.session_state.study_sessions = db.get_all_study_sessions()
                
                st.session_state.data_loaded = True
                
            except Exception as e:
                st.error(f"Erro ao carregar dados: {e}")
                # Inicializar com valores vazios em caso de erro
                st.session_state.notes = {}
                st.session_state.flashcards = {}
                st.session_state.quizzes = {}
                st.session_state.study_sessions = []
        
        # Inicializar outras variáveis de estado
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
        """Salva nota no banco e sessão"""
        st.session_state.notes[note.id] = note
        st.session_state.db_manager.save_note(note)
    
    @staticmethod
    def save_flashcard(flashcard):
        """Salva flashcard no banco e sessão"""
        st.session_state.flashcards[flashcard.id] = flashcard
        st.session_state.db_manager.save_flashcard(flashcard)
    
    @staticmethod
    def save_quiz(quiz):
        """Salva quiz no banco e sessão"""
        st.session_state.quizzes[quiz.id] = quiz
        st.session_state.db_manager.save_quiz(quiz)
    
    @staticmethod
    def save_study_session(session):
        """Salva sessão de estudo no banco e sessão"""
        st.session_state.study_sessions.append(session)
        st.session_state.db_manager.save_study_session(session)
    
    @staticmethod
    def save_api_key(api_key):
        """Salva chave da API no banco"""
        st.session_state.openai_api_key = api_key
        st.session_state.db_manager.save_config('openai_api_key', api_key)
    
    @staticmethod
    def delete_note(note_id):
        """Deleta nota do banco e sessão"""
        if note_id in st.session_state.notes:
            del st.session_state.notes[note_id]
            st.session_state.db_manager.delete_note(note_id)
    
    @staticmethod
    def delete_flashcard(flashcard_id):
        """Deleta flashcard do banco e sessão"""
        if flashcard_id in st.session_state.flashcards:
            del st.session_state.flashcards[flashcard_id]
            st.session_state.db_manager.delete_flashcard(flashcard_id)
    
    @staticmethod
    def delete_quiz(quiz_id):
        """Deleta quiz do banco e sessão"""
        if quiz_id in st.session_state.quizzes:
            del st.session_state.quizzes[quiz_id]
            st.session_state.db_manager.delete_quiz(quiz_id)


# ===============================
# PÁGINA DE CONFIGURAÇÕES DO BANCO
# ===============================

class DatabaseSettingsPage:
    """Página para gerenciar banco de dados"""
    
    @staticmethod
    def render():
        st.markdown("### 🗄️ Configurações do Banco de Dados")
        
        db = st.session_state.db_manager
        
        # Estatísticas
        col1, col2, col3 = st.columns(3)
        stats = db.get_statistics()
        
        with col1:
            st.metric("📝 Total de Notas", stats['total_notes'])
            st.metric("🗃️ Total de Flashcards", stats['total_flashcards'])
        
        with col2:
            st.metric("📊 Total de Quizzes", stats['total_quizzes'])
            st.metric("📚 Sessões de Estudo", stats['total_sessions'])
        
        with col3:
            st.metric("⏱️ Tempo Total", f"{stats['total_study_time']} min")
            size_mb = stats['database_size'] / 1024 / 1024
            st.metric("💾 Tamanho do Banco", f"{size_mb:.2f} MB")
        
        st.markdown("---")
        
        # Ações
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💾 Backup")
            if st.button("🔄 Criar Backup", use_container_width=True):
                if db.create_backup():
                    st.success("✅ Backup criado com sucesso!")
                else:
                    st.error("❌ Erro ao criar backup")
            
            if st.button("📥 Restaurar Backup", use_container_width=True):
                if db.restore_backup():
                    st.success("✅ Backup restaurado!")
                    st.rerun()
                else:
                    st.error("❌ Erro ao restaurar backup")
        
        with col2:
            st.markdown("#### 📤 Exportar/Importar")
            
            # Exportar
            export_filename = st.text_input("Nome do arquivo", "studyai_export.json")
            if st.button("📤 Exportar Dados", use_container_width=True):
                try:
                    db.export_to_json(export_filename)
                    st.success(f"✅ Dados exportados para {export_filename}")
                except Exception as e:
                    st.error(f"❌ Erro ao exportar: {e}")
            
            # Importar
            uploaded_file = st.file_uploader("Importar dados", type=['json'])
            if uploaded_file is not None:
                if st.button("📥 Importar", use_container_width=True):
                    try:
                        # Salvar temporariamente
                        temp_path = "temp_import.json"
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        db.import_from_json(temp_path)
                        os.remove(temp_path)
                        
                        st.success("✅ Dados importados com sucesso!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erro ao importar: {e}")

