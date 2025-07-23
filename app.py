import streamlit as st
import json
from datetime import datetime, timedelta
from datetime import date as current_date
import random
from typing import Dict, List, Any, Optional, Tuple
import base64
import io
import hashlib
import re
from dataclasses import dataclass, asdict
import numpy as np
# Importações principais

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from PIL import Image
#import easyocr
import pytesseract
#import cv2
import openai
from dotenv import load_dotenv
import os

from enhanced_session_manager import EnhancedSessionManager, DatabaseManager, DatabaseSettingsPage
from user_auth_system import (
    User, AuthenticationSystem, UserDatabaseManager,
    AuthPages, UserProfilePage, UserAwareSessionManager
)
from fixes import apply_all_fixes
load_dotenv()

# Configuração da página
st.set_page_config(
    page_title="StudyAI - Sistema de Estudos Inteligente",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# ESTILOS CSS
# ===============================

def load_css():
    """Carrega estilos CSS para a aplicação"""
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        
        .card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            color: white;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .note-card {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            transition: transform 0.2s;
        }
        
        .note-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .flashcard {
            background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }
        
        .stats-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.5rem;
            transition: transform 0.2s;
        }
        
        .stats-card:hover {
            transform: translateY(-3px);
        }
        
        .success-message {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .error-message {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        @media (max-width: 768px) {
            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
                padding-left: 0.5rem;
                padding-right: 0.5rem;
            }
            
            .flashcard {
                padding: 1.5rem;
                min-height: 150px;
                font-size: 1rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# CLASSES DE DADOS
# ===============================

@dataclass
class Note:
    """Classe para representar uma nota de estudo"""
    id: str
    title: str
    content: str
    tags: List[str]
    connections: List[str]
    created_at: str
    category: str
    
    def to_dict(self) -> Dict:
        """Converte a nota para dicionário"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Note':
        """Cria uma nota a partir de dicionário"""
        return cls(**data)

@dataclass
class Flashcard:
    """Classe para representar um flashcard"""
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
        """Calcula a precisão do flashcard"""
        if self.total_reviews == 0:
            return 0.0
        return (self.correct_count / self.total_reviews) * 100
    
    def to_dict(self) -> Dict:
        """Converte o flashcard para dicionário"""
        return asdict(self)

@dataclass
class Quiz:
    """Classe para representar um quiz"""
    id: str
    title: str
    questions: List[Dict]
    category: str
    created_at: str
    
    def to_dict(self) -> Dict:
        """Converte o quiz para dicionário"""
        return asdict(self)

@dataclass
class StudySession:
    """Classe para representar uma sessão de estudo"""
    date: str
    activity: str
    duration: int
    category: str
    score: float
    
    def to_dict(self) -> Dict:
        """Converte a sessão para dicionário"""
        return asdict(self)
class EmailService:
    """Serviço de email para recuperação de senha (exemplo)"""
    
    @staticmethod
    def send_reset_email(email: str, token: str, username: str):
        """Envia email com token de recuperação"""
        # Configurações do servidor SMTP (exemplo com Gmail)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "noreply@studyai.com"  # Substituir
        sender_password = "app_password"       # Usar App Password do Gmail
        
        # Criar mensagem
        message = MIMEMultipart("alternative")
        message["Subject"] = "StudyAI - Recuperação de Senha"
        message["From"] = sender_email
        message["To"] = email
        
        # Corpo do email
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #1f77b4;">🧠 StudyAI - Recuperação de Senha</h2>
                
                <p>Olá {username},</p>
                
                <p>Recebemos uma solicitação para redefinir sua senha.</p>
                
                <p>Use o token abaixo para criar uma nova senha:</p>
                
                <div style="background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <code style="font-size: 16px; font-weight: bold;">{token}</code>
                </div>
                
                <p><strong>Este token expira em 1 hora.</strong></p>
                
                <p>Se você não solicitou esta alteração, ignore este email.</p>
                
                <hr style="margin: 30px 0;">
                
                <p style="color: #666; font-size: 12px;">
                    StudyAI - Sistema de Estudos Inteligente<br>
                    Este é um email automático, não responda.
                </p>
            </div>
        </body>
        </html>
        """
        
        part = MIMEText(html, "html")
        message.attach(part)
        
        # Enviar email
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(message)
            return True
        except Exception as e:
            print(f"Erro ao enviar email: {e}")
            return False 
    
class ImprovedQuizGenerator:
    """Gerador de quiz melhorado com tratamento de erros robusto"""
    
    @staticmethod
    def generate_quiz(client, topic: str, num_questions: int = 5) -> List[Dict]:
        """Gera quiz com melhor tratamento de erros e validação"""
        try:
            prompt = f"""
            Crie um quiz sobre "{topic}" com {num_questions} questões de múltipla escolha.
            
            IMPORTANTE: Retorne APENAS um array JSON válido, sem texto adicional.
            
            Formato EXATO:
            [
                {{
                    "question": "pergunta clara",
                    "options": ["opção 1", "opção 2", "opção 3", "opção 4"],
                    "correct": 0,
                    "explanation": "explicação detalhada"
                }}
            ]
            
            Regras:
            - O campo "correct" deve ser um número de 0 a 3 (índice da resposta correta)
            - Exatamente 4 opções por questão
            - Todas as strings devem usar aspas duplas
            - Sem vírgulas após o último elemento
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um gerador de quiz educacional. Retorne APENAS JSON válido."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.5
            )
            
            content = response.choices[0].message.content.strip()
            
            # Múltiplas tentativas de extrair JSON
            json_data = None
            
            # Tentativa 1: Parse direto
            try:
                json_data = json.loads(content)
            except:
                # Tentativa 2: Extrair JSON com regex
                json_matches = re.findall(r'\[[\s\S]*\]', content)
                if json_matches:
                    try:
                        json_data = json.loads(json_matches[0])
                    except:
                        pass
            
            if not json_data:
                # Tentativa 3: Limpar e tentar novamente
                content_clean = content.replace("'", '"').replace('\n', ' ')
                content_clean = re.sub(r',\s*}', '}', content_clean)
                content_clean = re.sub(r',\s*]', ']', content_clean)
                
                try:
                    json_data = json.loads(content_clean)
                except:
                    raise ValueError("Não foi possível extrair JSON válido da resposta")
            
            # Validar estrutura do quiz
            validated_questions = []
            for q in json_data[:num_questions]:
                if ImprovedQuizGenerator._validate_question(q):
                    validated_questions.append(q)
            
            if not validated_questions:
                raise ValueError("Nenhuma questão válida foi gerada")
                
            return validated_questions
            
        except Exception as e:
            print(f"Erro na geração do quiz: {str(e)}")
            # Retornar quiz de fallback melhorado
            return ImprovedQuizGenerator._get_fallback_quiz(topic, num_questions)
    
    @staticmethod
    def _validate_question(question: Dict) -> bool:
        """Valida estrutura de uma questão"""
        required_fields = ['question', 'options', 'correct', 'explanation']
        
        # Verificar campos obrigatórios
        if not all(field in question for field in required_fields):
            return False
        
        # Validar tipos
        if not isinstance(question['question'], str):
            return False
        
        if not isinstance(question['options'], list) or len(question['options']) != 4:
            return False
        
        if not isinstance(question['correct'], int) or question['correct'] not in range(4):
            return False
        
        if not isinstance(question['explanation'], str):
            return False
            
        return True
    
    @staticmethod
    def _get_fallback_quiz(topic: str, num_questions: int) -> List[Dict]:
        """Quiz de fallback melhorado por categoria"""
        fallback_quizzes = {
            "default": [
                {
                    "question": f"Qual é a definição correta de {topic}?",
                    "options": [
                        f"Uma técnica avançada relacionada a {topic}",
                        f"Um conceito fundamental de {topic}",
                        f"Uma aplicação prática de {topic}",
                        f"Um método de análise de {topic}"
                    ],
                    "correct": 1,
                    "explanation": f"A definição correta envolve os conceitos fundamentais de {topic}."
                }
            ]
        }
        
        quiz = fallback_quizzes.get(topic.lower(), fallback_quizzes["default"])
        return quiz[:num_questions]
    
class AdvancedOCR:
    """Sistema OCR avançado com múltiplos engines"""
    
    def __init__(self):
        # Inicializar EasyOCR para português e inglês
        self.reader = None
        # try:
        #     self.reader = easyocr.Reader(['pt', 'en'], gpu=False)
        # except:
        #     print("EasyOCR não disponível")
    
    def extract_text_pytesseract(self, image) -> str:
        """Extrai texto usando Pytesseract"""
        try:
            # Converter PIL para OpenCV
            img_array = np.array(image)
            
            # Pré-processamento da imagem
            processed = self.preprocess_image(img_array)
            
            # Configurar Tesseract para português
            custom_config = r'--oem 3 --psm 6 -l por+eng'
            
            # Extrair texto
            text = pytesseract.image_to_string(processed, config=custom_config)
            
            return text.strip()
            
        except Exception as e:
            print(f"Erro no Pytesseract: {str(e)}")
            return ""
    
    def extract_text_easyocr(self, image) -> str:
        """Extrai texto usando EasyOCR"""
        if not self.reader:
            return ""
            
        try:
            # Converter PIL para array numpy
            img_array = np.array(image)
            
            # Executar OCR
            results = self.reader.readtext(img_array)
            
            # Concatenar todos os textos detectados
            text_parts = [result[1] for result in results]
            return " ".join(text_parts)
            
        except Exception as e:
            print(f"Erro no EasyOCR: {str(e)}")
            return ""
    
    def preprocess_image(self, image):
        """Pré-processa imagem para melhorar OCR"""
        # Converter para escala de cinza
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Redimensionar se muito pequena
        height, width = gray.shape
        if width < 1000:
            scale = 1000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Aplicar threshold adaptativo
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Remover ruído
        denoised = cv2.medianBlur(thresh, 3)
        
        # Aplicar dilatação e erosão para conectar texto
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def extract_text(self, image) -> str:
        """Método principal que tenta múltiplos engines"""
        # Tentar Pytesseract primeiro
        text_tesseract = self.extract_text_pytesseract(image)
        
        # Se o resultado for muito curto, tentar EasyOCR
        if len(text_tesseract) < 50 and self.reader:
            text_easyocr = self.extract_text_easyocr(image)
            
            # Retornar o texto mais longo
            return text_easyocr if len(text_easyocr) > len(text_tesseract) else text_tesseract
        
        return text_tesseract
    
    def extract_handwritten_text(self, image) -> str:
        """Especializado para texto manuscrito"""
        try:
            # Pré-processamento específico para manuscrito
            img_array = np.array(image)
            
            # Converter para escala de cinza
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array
            
            # Aumentar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Binarização
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR com configuração para manuscrito
            custom_config = r'--oem 3 --psm 6 -l por+eng'
            text = pytesseract.image_to_string(binary, config=custom_config)
            
            return text.strip()
            
        except Exception as e:
            print(f"Erro no OCR manuscrito: {str(e)}")
            return self.extract_text(image)  # Fallback para método padrão

# ===============================
# GERENCIADOR DE ESTADO
# ===============================

class SessionManager:
    """Gerencia o estado da sessão do Streamlit"""
    
    @staticmethod
    def init_session_state():
        """Inicializa todas as variáveis de estado"""
        defaults = {
            'notes': {},
            'flashcards': {},
            'quizzes': {},
            'study_sessions': [],
            'current_page': "Dashboard",
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
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
    def save_data():
        """Salva dados importantes (placeholder para futuro banco de dados)"""
        # TODO: Implementar salvamento em banco de dados
        pass
    
    @staticmethod
    def load_data():
        """Carrega dados salvos (placeholder para futuro banco de dados)"""
        # TODO: Implementar carregamento de banco de dados
        pass

# ===============================
# MÓDULO 3: SISTEMA DE PROGRESSO AVANÇADO
# ===============================

@dataclass
class StudyMetrics:
    """Métricas detalhadas de estudo"""
    total_study_time: int
    average_session_duration: float
    study_streak: int
    best_category: str
    improvement_rate: float
    consistency_score: float
    mastery_level: Dict[str, float]
    weekly_goals_met: int
    study_patterns: Dict[str, float]

class AdvancedProgressTracker:
    """Sistema avançado de tracking de progresso"""
    
    def __init__(self, study_sessions: List, flashcards: Dict, quizzes: Dict):
        self.sessions = study_sessions
        self.flashcards = flashcards
        self.quizzes = quizzes
    
    def calculate_comprehensive_metrics(self) -> StudyMetrics:
        """Calcula métricas abrangentes de estudo"""
        if not self.sessions:
            return self._get_empty_metrics()
        
        df = pd.DataFrame([s for s in self.sessions])
        df['date'] = pd.to_datetime(df['date'])
        
        # Métricas básicas
        total_time = df['duration'].sum()
        avg_duration = df['duration'].mean()
        
        # Streak de estudo
        study_streak = self._calculate_study_streak(df)
        
        # Melhor categoria
        category_performance = df.groupby('category')['score'].mean()
        best_category = category_performance.idxmax() if not category_performance.empty else "N/A"
        
        # Taxa de melhoria
        improvement_rate = self._calculate_improvement_rate(df)
        
        # Score de consistência
        consistency_score = self._calculate_consistency_score(df)
        
        # Nível de maestria por categoria
        mastery_level = self._calculate_mastery_levels(df)
        
        # Metas semanais
        weekly_goals = self._check_weekly_goals(df)
        
        # Padrões de estudo
        study_patterns = self._analyze_study_patterns(df)
        
        return StudyMetrics(
            total_study_time=int(total_time),
            average_session_duration=round(avg_duration, 1),
            study_streak=study_streak,
            best_category=best_category,
            improvement_rate=improvement_rate,
            consistency_score=consistency_score,
            mastery_level=mastery_level,
            weekly_goals_met=weekly_goals,
            study_patterns=study_patterns
        )
    
    def _calculate_study_streak(self, df: pd.DataFrame) -> int:
        """Calcula sequência atual de dias estudando"""
        dates = df['date'].dt.date.unique()
        dates = sorted(dates, reverse=True)
        
        today = datetime.now().date()
        streak = 0
        
        for i in range(len(dates)):
            expected_date = today - timedelta(days=i)
            if i < len(dates) and dates[i] == expected_date:
                streak += 1
            else:
                break
                
        return streak
    
    def _calculate_improvement_rate(self, df: pd.DataFrame) -> float:
        """Calcula taxa de melhoria ao longo do tempo"""
        if len(df) < 2:
            return 0.0
        
        # Agrupar por semana e calcular média
        df['week'] = df['date'].dt.isocalendar().week
        weekly_scores = df.groupby('week')['score'].mean()
        
        if len(weekly_scores) < 2:
            return 0.0
        
        # Calcular tendência
        x = list(range(len(weekly_scores)))
        y = weekly_scores.values
        
        # Regressão linear simples
        n = len(x)
        if n == 0:
            return 0.0
            
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i]**2 for i in range(n))
        
        if (n * sum_x2 - sum_x**2) == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        
        return round(slope * 100, 1)  # Percentual de melhoria
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calcula score de consistência (0-100)"""
        if len(df) == 0:
            return 0.0
        
        # Dias únicos de estudo
        study_days = df['date'].dt.date.nunique()
        
        # Período total (primeira à última sessão)
        date_range = (df['date'].max() - df['date'].min()).days + 1
        
        if date_range == 0:
            return 100.0
        
        # Consistência = dias estudados / dias totais * 100
        consistency = (study_days / date_range) * 100
        
        # Bonus por regularidade
        avg_gap = date_range / study_days if study_days > 0 else float('inf')
        regularity_bonus = max(0, 20 - avg_gap * 2)
        
        return min(100, round(consistency + regularity_bonus, 1))
    
    def _calculate_mastery_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula nível de domínio por categoria"""
        mastery = {}
        
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            
            # Fatores de maestria
            avg_score = cat_data['score'].mean()
            total_time = cat_data['duration'].sum()
            sessions = len(cat_data)
            
            # Fórmula de maestria
            time_factor = min(1.0, total_time / 300)  # 300 min = máximo
            session_factor = min(1.0, sessions / 10)   # 10 sessões = máximo
            score_factor = avg_score
            
            mastery_score = (score_factor * 0.5 + time_factor * 0.3 + session_factor * 0.2) * 100
            mastery[category] = round(mastery_score, 1)
        
        return mastery
    
    def _check_weekly_goals(self, df: pd.DataFrame) -> int:
        """Verifica quantas semanas as metas foram cumpridas"""
        # Meta: 3 sessões por semana, 60 min total
        weekly_data = df.groupby(df['date'].dt.isocalendar().week).agg({
            'duration': ['sum', 'count']
        })
        
        goals_met = 0
        for week_stats in weekly_data.values:
            total_duration = week_stats[0]
            session_count = week_stats[1]
            
            if total_duration >= 60 and session_count >= 3:
                goals_met += 1
        
        return goals_met
    
    def _analyze_study_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analisa padrões de estudo (horários preferenciais)"""
        df['hour'] = df['date'].dt.hour
        
        # Distribuição por período do dia
        patterns = {
            'morning': len(df[(df['hour'] >= 6) & (df['hour'] < 12)]),
            'afternoon': len(df[(df['hour'] >= 12) & (df['hour'] < 18)]),
            'evening': len(df[(df['hour'] >= 18) & (df['hour'] < 24)]),
            'night': len(df[(df['hour'] >= 0) & (df['hour'] < 6)])
        }
        
        total = sum(patterns.values())
        if total > 0:
            patterns = {k: round(v/total * 100, 1) for k, v in patterns.items()}
        
        return patterns
    
    def _get_empty_metrics(self) -> StudyMetrics:
        """Retorna métricas vazias"""
        return StudyMetrics(
            total_study_time=0,
            average_session_duration=0.0,
            study_streak=0,
            best_category="N/A",
            improvement_rate=0.0,
            consistency_score=0.0,
            mastery_level={},
            weekly_goals_met=0,
            study_patterns={}
        )
    
    def generate_advanced_visualizations(self) -> Dict[str, go.Figure]:
        """Gera visualizações avançadas"""
        if not self.sessions:
            return {}
        
        df = pd.DataFrame([s for s in self.sessions])
        df['date'] = pd.to_datetime(df['date'])
        
        figures = {}
        
        # 1. Heatmap de atividade
        figures['heatmap'] = self._create_activity_heatmap(df)
        
        # 2. Radar chart de habilidades
        figures['radar'] = self._create_skills_radar(df)
        
        # 3. Gráfico de progresso cumulativo
        figures['cumulative'] = self._create_cumulative_progress(df)
        
        # 4. Análise de performance por dia da semana
        figures['weekday'] = self._create_weekday_analysis(df)
        
        return figures
    
    def _create_activity_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Cria heatmap de atividade estilo GitHub"""
        # Preparar dados
        df['weekday'] = df['date'].dt.weekday
        df['week'] = df['date'].dt.isocalendar().week
        
        # Agregar por dia
        daily_data = df.groupby(['week', 'weekday'])['duration'].sum().reset_index()
        
        # Criar matriz
        weeks = sorted(daily_data['week'].unique())
        heatmap_data = []
        
        for week in weeks:
            week_data = []
            for day in range(7):
                value = daily_data[(daily_data['week'] == week) & 
                                 (daily_data['weekday'] == day)]['duration'].sum()
                week_data.append(value)
            heatmap_data.append(week_data)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'],
            y=[f'Semana {w}' for w in weeks],
            colorscale='Viridis',
            hovertemplate='<b>Dia:</b> %{x}<br>' +
                  '<b>Semana:</b> %{y}<br>' +
                  '<b>Duração:</b> %{z} minutos<extra></extra>'
        ))
        
        fig.update_layout(
            title='Mapa de Calor - Atividade de Estudo',
            xaxis_title='Dia da Semana',
            yaxis_title='Semana',
            height=400
        )
        
        return fig
    
    def _create_skills_radar(self, df: pd.DataFrame) -> go.Figure:
        """Cria radar chart de habilidades por categoria"""
        metrics = self.calculate_comprehensive_metrics()
        
        categories = list(metrics.mastery_level.keys())
        values = list(metrics.mastery_level.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Nível de Domínio'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Radar de Habilidades por Categoria"
        )
        
        return fig
    
    def _create_cumulative_progress(self, df: pd.DataFrame) -> go.Figure:
        """Cria gráfico de progresso cumulativo"""
        df_sorted = df.sort_values('date')
        df_sorted['cumulative_time'] = df_sorted['duration'].cumsum()
        df_sorted['cumulative_sessions'] = range(1, len(df_sorted) + 1)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Tempo Total Acumulado', 'Sessões Acumuladas'),
            shared_xaxes=True
        )
        
        # Tempo acumulado
        fig.add_trace(
            go.Scatter(
                x=df_sorted['date'],
                y=df_sorted['cumulative_time'],
                mode='lines+markers',
                name='Tempo (min)',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Sessões acumuladas
        fig.add_trace(
            go.Scatter(
                x=df_sorted['date'],
                y=df_sorted['cumulative_sessions'],
                mode='lines+markers',
                name='Sessões',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Data", row=2, col=1)
        fig.update_yaxes(title_text="Minutos", row=1, col=1)
        fig.update_yaxes(title_text="Quantidade", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        
        return fig
    
    def _create_weekday_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Análise de performance por dia da semana"""
        df['weekday'] = df['date'].dt.day_name()
        
        weekday_stats = df.groupby('weekday').agg({
            'score': 'mean',
            'duration': 'mean'
        }).round(2)
        
        # Ordenar dias da semana
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday']
        weekday_stats = weekday_stats.reindex(weekday_order)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Pontuação Média',
            x=weekday_stats.index,
            y=weekday_stats['score'] * 100,
            yaxis='y',
            offsetgroup=1
        ))
        
        fig.add_trace(go.Bar(
            name='Duração Média (min)',
            x=weekday_stats.index,
            y=weekday_stats['duration'],
            yaxis='y2',
            offsetgroup=2
        ))
        
        fig.update_layout(
            title='Análise de Performance por Dia da Semana',
            xaxis=dict(title='Dia da Semana'),
            yaxis=dict(title='Pontuação (%)', side='left'),
            yaxis2=dict(title='Duração (min)', overlaying='y', side='right'),
            barmode='group'
        )
        
        return fig
# ===============================
# INTEGRAÇÃO COM OPENAI
# ===============================

class OpenAIService:
    """Serviço para integração com OpenAI API"""
    
    # @staticmethod
    # def get_client() -> Optional[openai.OpenAI]:
    #     """Obtém cliente OpenAI configurado"""
    #     api_key = st.session_state.get('openai_api_key')
    #     api_key = os.getenv('OPENAI_API_KEY')  # Prioriza variável de ambiente
    #     UserAwareSessionManager.save_api_key#EnhancedSessionManager.save_api_key(api_key)
    #     if not api_key:
    #         return None
    #     try:
    #         #EnhancedSessionManager.save_api_key(api_key)
    #         return openai.OpenAI(api_key=api_key)
    #     except Exception as e:
    #         st.error(f"Erro ao configurar OpenAI: {str(e)}")
    #         return None
    @staticmethod
    def get_client():
        load_dotenv()  # Carrega .env
        
        # Prioridade: .env > session_state > user_config
        api_key = os.getenv('OPENAI_API_KEY') or st.session_state.get('openai_api_key')
        
        if api_key:
            st.session_state.openai_api_key = api_key  # Sincroniza
            UserAwareSessionManager.save_api_key(api_key)  # Salva na sessão
            return openai.OpenAI(api_key=api_key)
        return None

    @staticmethod
    def test_api_key(api_key: str) -> bool:
        """Testa se a chave da API é válida"""
        try:
            client = openai.OpenAI(api_key=api_key)
            # Teste simples com uma requisição mínima
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception:
            return False
    
    @classmethod
    def generate_insights(cls, notes_content: str) -> str:
        """Gera insights sobre as notas usando GPT"""
        client = cls.get_client()
        if not client:
            return "⚠️ Configure sua chave da OpenAI para gerar insights personalizados."
        
        try:
            # Limita o conteúdo para evitar custos altos
            content = notes_content[:3000] if len(notes_content) > 3000 else notes_content
            
            prompt = f"""
            Analise o seguinte conteúdo de notas de estudo e gere insights valiosos em português:

            {content}

            Forneça exatamente 4 insights no formato:
            🔍 **Padrão identificado**: [sua análise]
            💡 **Sugestão de estudo**: [sua sugestão] 
            🎯 **Lacuna detectada**: [lacuna identificada]
            📈 **Progresso observado**: [avaliação do progresso]

            Seja específico e prático. Use markdown para formatação.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Erro ao gerar insights: {str(e)}"
    
    @classmethod
    def generate_flashcards(cls, content: str, num_cards: int = 5) -> List[Dict]:
        """Gera flashcards automaticamente"""
        client = cls.get_client()
        if not client:
            return cls._get_fallback_flashcards(num_cards)
        
        try:
            content = content[:2000] if len(content) > 2000 else content
            
            prompt = f"""
            Com base no seguinte conteúdo, crie exatamente {num_cards} flashcards educativos em português.

            Conteúdo: {content}

            RETORNE APENAS um array JSON válido no formato:
            [{{"q": "pergunta clara e específica", "a": "resposta completa e educativa"}}]

            Requisitos:
            - Perguntas devem testar compreensão real
            - Respostas devem ser precisas e educativas
            - Use linguagem clara e objetiva
            - Foque nos conceitos mais importantes
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.5
            )
            
            # Extrai JSON da resposta
            content_response = response.choices[0].message.content
            json_match = re.search(r'\[.*\]', content_response, re.DOTALL)
            
            if json_match:
                flashcards_data = json.loads(json_match.group())
                return flashcards_data[:num_cards]
            else:
                raise ValueError("Resposta não contém JSON válido")
                
        except Exception as e:
            st.error(f"Erro ao gerar flashcards com IA: {str(e)}")
            return cls._get_fallback_flashcards(num_cards)
    
    @classmethod
    def generate_quiz(cls, topic: str, num_questions: int = 5) -> List[Dict]:
        """Gera quiz automaticamente"""
        client = cls.get_client()
        if not client:
            return cls._get_fallback_quiz(topic)
        
        try:
            prompt = f"""
            Crie um quiz educativo sobre "{topic}" com exatamente {num_questions} questões múltipla escolha em português.

            RETORNE APENAS um array JSON válido no formato:
            [{{
                "question": "pergunta clara e desafiadora",
                "options": ["opção 1", "opção 2", "opção 3", "opção 4"],
                "correct": 0,
                "explanation": "explicação detalhada da resposta correta"
            }}]

            Requisitos:
            - Questões de nível intermediário a avançado
            - 4 opções por questão, apenas 1 correta
            - Opções plausíveis mas distintas
            - Explicações educativas
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.5
            )
            
            content_response = response.choices[0].message.content
            json_match = re.search(r'\[.*\]', content_response, re.DOTALL)
            
            if json_match:
                quiz_data = json.loads(json_match.group())
                return ImprovedQuizGenerator.generate_quiz(client, topic, num_questions) #quiz_data[:num_questions]
            else:
                raise ValueError("Resposta não contém JSON válido")
                
        except Exception as e:
            st.error(f"Erro ao gerar quiz com IA: {str(e)}")
            return cls._get_fallback_quiz(topic)
    
    @staticmethod
    def _get_fallback_flashcards(num_cards: int) -> List[Dict]:
        """Flashcards de exemplo quando a API falha"""
        examples = [
            {"q": "O que é inteligência artificial?", "a": "Campo da ciência da computação que visa criar sistemas capazes de realizar tarefas que normalmente requerem inteligência humana."},
            {"q": "Defina machine learning", "a": "Subcampo da IA que permite aos computadores aprender e melhorar automaticamente através da experiência."},
            {"q": "O que são redes neurais?", "a": "Modelos computacionais inspirados no funcionamento do cérebro humano."},
            {"q": "Explique overfitting", "a": "Fenômeno onde um modelo se ajusta excessivamente aos dados de treinamento."},
            {"q": "O que é validação cruzada?", "a": "Técnica para avaliar a capacidade de generalização de um modelo."},
        ]
        return examples[:num_cards]
    
    @staticmethod
    def _get_fallback_quiz(topic: str) -> List[Dict]:
        """Quiz de exemplo quando a API falha"""
        return [{
            "question": f"Esta é uma questão de exemplo sobre {topic}. Configure sua chave OpenAI para gerar questões reais.",
            "options": ["Configure OpenAI", "Exemplo A", "Exemplo B", "Exemplo C"],
            "correct": 0,
            "explanation": "Configure sua chave da OpenAI API para gerar quizzes personalizados automaticamente."
        }]

# ===============================
# UTILITÁRIOS
# ===============================

class Utils:
    """Funções utilitárias diversas"""
    
    @staticmethod
    def generate_id(content: str) -> str:
        """Gera ID único baseado no conteúdo e timestamp"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(f"{content}{timestamp}".encode()).hexdigest()
    
    @staticmethod
    def format_date(date_string: str) -> str:
        """Formata data para exibição"""
        try:
            date_obj = datetime.fromisoformat(date_string)
            return date_obj.strftime("%d/%m/%Y %H:%M")
        except:
            return date_string
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 50) -> str:
        """Trunca texto mantendo legibilidade"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    @staticmethod
    def extract_text_from_image(image) -> str:
        """Simula OCR - substitua por implementação real"""
        ocr = AdvancedOCR()
        return ocr.extract_text(image)

# ===============================
# COMPONENTES DA INTERFACE
# ===============================

class UIComponents:
    """Componentes reutilizáveis da interface"""
    
    @staticmethod
    def show_api_config():
        """Mostra configuração da API OpenAI"""
        if not st.session_state.get('openai_api_key'):
            st.warning("⚠️ Configure sua chave da OpenAI para funcionalidades completas de IA")
            with st.expander("⚙️ Configurar OpenAI API"):
                api_key = st.text_input(
                    "Chave da API OpenAI", 
                    type="password", 
                    help="Cole sua chave da API da OpenAI aqui"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Salvar Chave"):
                        if api_key:
                            if OpenAIService.test_api_key(api_key):
                                UserAwareSessionManager.save_api_key(api_key) #st.session_state.openai_api_key = api_key
                                st.success("✅ Chave salva e validada!")
                                st.rerun()
                            else:
                                st.error("❌ Chave inválida. Verifique e tente novamente.")
                        else:
                            st.error("❌ Digite uma chave válida")
                
                with col2:
                    if st.button("🔗 Obter Chave OpenAI"):
                        st.info("Visite: https://platform.openai.com/api-keys")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success("✅ OpenAI API configurada!")
            with col2:
                if st.button("🔄 Reconfigurar"):
                    del st.session_state.openai_api_key
                    st.rerun()
    
    @staticmethod
    def mobile_navigation():
        """Navegação otimizada para mobile"""
        pages = ["🏠 Dashboard", "📝 Notas", "🗃️ Flashcards", "📊 Simulados", "📁 Upload", "📈 Progresso"]
        
        cols = st.columns(len(pages))
        for i, page in enumerate(pages):
            with cols[i]:
                page_name = page.split(" ", 1)[1]
                is_current = st.session_state.current_page == page_name
                
                if st.button(
                    page_name, 
                    key=f"nav_{i}", 
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):
                    st.session_state.current_page = page_name
                    st.rerun()
    
    @staticmethod
    def stats_cards():
        """Mostra cards de estatísticas"""
        col1, col2, col3, col4 = st.columns(4)
        
        stats = [
            ("📝", len(st.session_state.notes), "Notas"),
            ("🗃️", len(st.session_state.flashcards), "Flashcards"),
            ("📊", len(st.session_state.quizzes), "Simulados"),
            ("⏱️", len(st.session_state.study_sessions), "Sessões")
        ]
        
        for col, (icon, value, label) in zip([col1, col2, col3, col4], stats):
            with col:
                st.markdown(f"""
                <div class='stats-card'>
                    <h3>{icon}</h3>
                    <h4>{value}</h4>
                    <p>{label}</p>
                </div>
                """, unsafe_allow_html=True)

# ===============================
# SISTEMA DE GRAFOS
# ===============================

class KnowledgeGraph:
    """Sistema de grafo de conhecimento"""
    
    @staticmethod
    def create_graph():
        """Cria e exibe o grafo de conhecimento"""
        if not st.session_state.notes:
            st.info("📝 Crie algumas notas primeiro para visualizar o grafo!")
            return
        
        # Criar grafo
        G = nx.Graph()
        
        # Adicionar nós
        for note_id, note in st.session_state.notes.items():
            G.add_node(note_id, 
                      title=note.title, 
                      category=note.category,
                      content_length=len(note.content),
                      tags=len(note.tags))
        
        # Adicionar arestas
        for note_id, note in st.session_state.notes.items():
            for connection_id in note.connections:
                if connection_id in st.session_state.notes:
                    G.add_edge(note_id, connection_id)
        
        # Criar visualização
        KnowledgeGraph._create_plotly_graph(G)
        KnowledgeGraph._show_graph_stats(G)
        KnowledgeGraph._show_graph_analysis(G)
    
    @staticmethod
    def _create_plotly_graph(G):
        """Cria visualização Plotly do grafo"""
        if len(G.nodes()) == 0:
            st.info("📝 Nenhuma nota encontrada!")
            return
        
        # Layout do grafo
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Preparar arestas
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Preparar nós
        node_x, node_y, node_text, node_info, node_colors, node_sizes = [], [], [], [], [], []
        
        category_colors = {
            'IA': '#FF6B6B',
            'Machine Learning': '#4ECDC4', 
            'Programação': '#45B7D1',
            'Matemática': '#96CEB4',
            'Outros': '#FECA57'
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            note = st.session_state.notes[node]
            node_text.append(Utils.truncate_text(note.title, 20))
            
            connections_count = len(note.connections)
            node_info.append(
                f"<b>{note.title}</b><br>"
                f"Categoria: {note.category}<br>"
                f"Tags: {len(note.tags)}<br>"
                f"Conexões: {connections_count}<br>"
                f"Tamanho: {len(note.content)} chars"
            )
            
            node_colors.append(category_colors.get(note.category, '#FECA57'))
            
            size = 20 + (connections_count * 10) + (len(note.content) / 100)
            node_sizes.append(min(size, 60))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=node_info,
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        # Criar figura
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text="🕸️ Grafo de Conhecimento - Suas Notas Conectadas",
                    x=0.5, 
                    font=dict(size=20)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="💡 Dica: Nós maiores = mais conexões ou conteúdo",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="#666", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    @staticmethod
    def _show_graph_stats(G):
        """Mostra estatísticas do grafo"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Total de Notas", len(G.nodes()))
        
        with col2:
            st.metric("🔗 Total de Conexões", len(G.edges()))
        
        with col3:
            if st.session_state.notes:
                avg_connections = sum([len(note.connections) for note in st.session_state.notes.values()]) / len(st.session_state.notes)
                st.metric("📊 Conexões/Nota", f"{avg_connections:.1f}")
            else:
                st.metric("📊 Conexões/Nota", "0.0")
        
        with col4:
            categories = len(set([note.category for note in st.session_state.notes.values()]))
            st.metric("🏷️ Categorias", categories)
    
    @staticmethod
    def _show_graph_analysis(G):
        """Mostra análise do grafo"""
        if len(G.nodes()) <= 1:
            return
        
        st.markdown("### 📊 Análise do Grafo")
        
        # Nota mais conectada
        connections_count = {note_id: len(note.connections) for note_id, note in st.session_state.notes.items()}
        if connections_count:
            most_connected = max(connections_count.items(), key=lambda x: x[1])
            
            if most_connected[1] > 0:
                most_connected_title = st.session_state.notes[most_connected[0]].title
                st.success(f"🌟 **Nota mais conectada:** {most_connected_title} ({most_connected[1]} conexões)")
        
        # Notas isoladas
        isolated_notes = [note_id for note_id, count in connections_count.items() if count == 0]
        if isolated_notes:
            st.warning(f"⚠️ **Notas isoladas:** {len(isolated_notes)} notas sem conexões")
        
        # Sugestões de conexões
        KnowledgeGraph._show_connection_suggestions()
    
    @staticmethod
    def _show_connection_suggestions():
        """Mostra sugestões de conexões"""
        st.markdown("### 💡 Sugestões de Conexões")
        
        suggestions = []
        notes = st.session_state.notes
        
        for note1_id, note1 in notes.items():
            for note2_id, note2 in notes.items():
                if note1_id != note2_id and note2_id not in note1.connections:
                    # Calcular similaridade
                    common_tags = set(note1.tags) & set(note2.tags)
                    same_category = note1.category == note2.category
                    
                    if len(common_tags) > 0 or same_category:
                        score = len(common_tags) + (2 if same_category else 0)
                        suggestions.append((note1.title, note2.title, note1_id, note2_id, score))
        
        # Mostrar top 3 sugestões
        suggestions.sort(key=lambda x: x[4], reverse=True)
        for i, (title1, title2, id1, id2, score) in enumerate(suggestions[:3]):
            if st.button(
                f"🔗 Conectar: {Utils.truncate_text(title1, 25)} ↔ {Utils.truncate_text(title2, 25)}", 
                key=f"suggest_{i}"
            ):
                st.session_state.notes[id1].connections.append(id2)
                st.success("✅ Conexão criada!")
                st.rerun()

# ===============================
# PÁGINAS PRINCIPAIS
# ===============================

class Dashboard:
    """Página principal do dashboard"""
    
    @staticmethod
    def render():
        st.markdown("<h1 class='main-header'>🧠 StudyAI Dashboard</h1>", unsafe_allow_html=True)
        
        # Configuração da API
        UIComponents.show_api_config()
        
        # Cards de estatísticas
        UIComponents.stats_cards()
        
        # Insights de IA
        Dashboard._render_ai_insights()
        
        # Atividade recente
        Dashboard._render_recent_activity()
    
    @staticmethod
    def _render_ai_insights():
        """Renderiza seção de insights da IA"""
        if len(st.session_state.notes) > 0:
            st.markdown("### 🤖 Insights da IA")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🔄 Gerar Novos Insights", use_container_width=True):
                    with st.spinner("🧠 Analisando suas notas..."):
                        all_notes_content = " ".join([note.content for note in st.session_state.notes.values()])
                        insights = OpenAIService.generate_insights(all_notes_content)
                        st.session_state.latest_insights = insights
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Limpar", use_container_width=True):
                    if 'latest_insights' in st.session_state:
                        del st.session_state.latest_insights
                        st.rerun()
            
            # Mostrar insights
            if 'latest_insights' in st.session_state:
                st.markdown(f"<div class='card'>{st.session_state.latest_insights}</div>", unsafe_allow_html=True)
            else:
                st.info("🤖 Clique em 'Gerar Novos Insights' para análise personalizada!")
    
    @staticmethod
    def _render_recent_activity():
        """Renderiza atividade recente"""
        st.markdown("### 📈 Atividade Recente")
        
        if st.session_state.study_sessions:
            # Últimas 5 sessões
            recent_sessions = st.session_state.study_sessions[-5:]
            df = pd.DataFrame([session for session in recent_sessions])
            
            fig = px.bar(
                df, 
                x='date', 
                y='duration', 
                color='activity', 
                title="Últimas Sessões de Estudo",
                labels={'duration': 'Duração (min)', 'date': 'Data'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📚 Nenhuma atividade registrada ainda. Comece a estudar!")

class NotesPage:
    """Página do sistema de notas"""
    
    @staticmethod
    def render():
        st.markdown("### 📝 Sistema de Notas Inteligente")
        
        tab1, tab2, tab3, tab4 = st.tabs(["✏️ Nova Nota", "🔍 Explorar", "🔗 Conexões", "🕸️ Grafo"])
        
        with tab1:
            NotesPage._render_new_note()
        
        with tab2:
            NotesPage._render_notes_explorer()
        
        with tab3:
            NotesPage._render_connections()
        
        with tab4:
            st.subheader("Grafo de Conhecimento")
            KnowledgeGraph.create_graph()
    
    @staticmethod
    def _render_new_note():
        """Renderiza formulário de nova nota"""
        st.subheader("Criar Nova Nota")
        
        with st.form("new_note_form", clear_on_submit=True):
            title = st.text_input("Título da nota*", placeholder="Digite o título da nota")
            category = st.selectbox("Categoria", ["IA", "Machine Learning", "Programação", "Matemática", "Outros"])
            content = st.text_area("Conteúdo*", height=200, placeholder="Digite o conteúdo da nota")
            tags = st.text_input("Tags (separadas por vírgula)", placeholder="ia, estudo, conceitos")
            
            submitted = st.form_submit_button("💾 Salvar Nota", use_container_width=True)
            
            if submitted:
                if title and content:
                    note_id = Utils.generate_id(title)
                    
                    note = Note(
                        id=note_id,
                        title=title.strip(),
                        content=content.strip(),
                        tags=[tag.strip() for tag in tags.split(",") if tag.strip()],
                        connections=[],
                        created_at=datetime.now().isoformat(),
                        category=category
                    )
                    
                    UserAwareSessionManager.save_note(note) #EnhancedSessionManager.save_note(note) #st.session_state.notes[note_id] = note
                    st.success("✅ Nota salva com sucesso!")
                    st.rerun()
                else:
                    st.error("❌ Preencha o título e conteúdo obrigatórios")
    
    @staticmethod
    def _render_notes_explorer():
        """Renderiza explorador de notas"""
        st.subheader("Suas Notas")
        
        if not st.session_state.notes:
            st.info("📝 Nenhuma nota ainda. Crie sua primeira nota!")
            return
        
        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            search = st.text_input("🔍 Buscar notas", placeholder="Digite para buscar...")
        
        with col2:
            categories = ["Todas"] + list(set([note.category for note in st.session_state.notes.values()]))
            category_filter = st.selectbox("Filtrar por categoria", categories)
        
        # Filtrar notas
        filtered_notes = []
        for note in st.session_state.notes.values():
            # Filtro de busca
            search_match = (
                not search or 
                search.lower() in note.title.lower() or 
                search.lower() in note.content.lower() or
                any(search.lower() in tag.lower() for tag in note.tags)
            )
            
            # Filtro de categoria
            category_match = category_filter == "Todas" or note.category == category_filter
            
            if search_match and category_match:
                filtered_notes.append(note)
        
        # Ordenar por data de criação (mais recente primeiro)
        filtered_notes.sort(key=lambda x: x.created_at, reverse=True)
        
        # Mostrar notas
        st.write(f"Encontradas: {len(filtered_notes)} notas")
        
        for note in filtered_notes:
            with st.expander(f"📝 {note.title} ({note.category})"):
                st.write(note.content)
                
                col1, col2 = st.columns(2)
                with col1:
                    if note.tags:
                        st.write(f"**Tags:** {', '.join(note.tags)}")
                    st.write(f"**Criada em:** {Utils.format_date(note.created_at)}")
                    if note.connections:
                        st.write(f"**Conexões:** {len(note.connections)}")
                
                with col2:
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        if st.button(f"🤖 Gerar Cards", key=f"flash_{note.id}"):
                            with st.spinner("Gerando flashcards..."):
                                generated_cards = OpenAIService.generate_flashcards(note.content, 3)
                                
                                for card_data in generated_cards:
                                    card_id = Utils.generate_id(card_data['q'])
                                    
                                    flashcard = Flashcard(
                                        id=card_id,
                                        question=card_data['q'],
                                        answer=card_data['a'],
                                        difficulty=3,
                                        category=note.category,
                                        last_reviewed="",
                                        next_review="",
                                        correct_count=0,
                                        total_reviews=0
                                    )
                                    
                                    UserAwareSessionManager.save_flashcard(flashcard) #EnhancedSessionManager.save_flashcard(flashcard) #st.session_state.flashcards[card_id] = flashcard
                                
                                st.success(f"✅ {len(generated_cards)} flashcards gerados!")
                    
                    with subcol2:
                        if st.button(f"🗑️ Excluir", key=f"del_{note.id}"):
                            del st.session_state.notes[note.id]
                            # Remover conexões que apontam para esta nota
                            for other_note in st.session_state.notes.values():
                                if note.id in other_note.connections:
                                    other_note.connections.remove(note.id)
                                    EnhancedSessionManager.delete_note(note.id)
                            st.rerun()
    
    @staticmethod
    def _render_connections():
        """Renderiza sistema de conexões"""
        st.subheader("Gerenciar Conexões")
        
        if len(st.session_state.notes) < 2:
            st.info("🔗 Crie pelo menos 2 notas para fazer conexões!")
            return
        
        # Criar nova conexão
        st.markdown("#### Criar Nova Conexão")
        
        note_titles = {note.id: note.title for note in st.session_state.notes.values()}
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            source_note = st.selectbox(
                "Nota de origem", 
                options=list(note_titles.keys()), 
                format_func=lambda x: note_titles[x]
            )
        
        with col2:
            available_targets = [k for k in note_titles.keys() if k != source_note]
            target_note = st.selectbox(
                "Conectar com", 
                options=available_targets, 
                format_func=lambda x: note_titles[x]
            )
        
        with col3:
            if st.button("🔗 Conectar", use_container_width=True):
                if target_note not in st.session_state.notes[source_note].connections:
                    st.session_state.notes[source_note].connections.append(target_note)
                    st.success("✅ Conexão criada!")
                    st.rerun()
                else:
                    st.warning("⚠️ Conexão já existe!")
        
        # Mostrar conexões existentes
        st.markdown("#### Conexões Existentes")
        
        connections_exist = False
        for note in st.session_state.notes.values():
            if note.connections:
                connections_exist = True
                st.write(f"📝 **{note.title}** conecta com:")
                for conn_id in note.connections:
                    if conn_id in st.session_state.notes:
                        connected_note = st.session_state.notes[conn_id]
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"   🔗 {connected_note.title}")
                        with col2:
                            if st.button("❌", key=f"remove_{note.id}_{conn_id}"):
                                note.connections.remove(conn_id)
                                st.rerun()
        
        if not connections_exist:
            st.info("📝 Nenhuma conexão criada ainda.")

# Continua na próxima parte devido ao limite de caracteres...
class FlashcardsByTheme:
    """Módulo para criar flashcards apenas com tema, sem depender de notas"""
    
    @staticmethod
    def render_theme_flashcards():
        """Interface para criar flashcards por tema"""
        st.markdown("### 🎯 Criar Flashcards por Tema")
        
        with st.form("theme_flashcards_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                theme = st.text_input(
                    "Digite o tema desejado*",
                    placeholder="Ex: Inteligência Artificial, Python, Matemática",
                    help="O sistema vai gerar flashcards sobre este tema"
                )
            
            with col2:
                num_cards = st.number_input(
                    "Quantidade",
                    min_value=1,
                    max_value=20,
                    value=5
                )
            
            difficulty = st.select_slider(
                "Nível de dificuldade",
                options=["Iniciante", "Intermediário", "Avançado"],
                value="Intermediário"
            )
            
            category = st.selectbox(
                "Categoria",
                ["IA", "Machine Learning", "Programação", "Matemática", "Outros"]
            )
            
            submitted = st.form_submit_button("🤖 Gerar Flashcards", use_container_width=True)
            
            if submitted and theme:
                FlashcardsByTheme._generate_theme_flashcards(
                    theme, num_cards, difficulty, category
                )
            elif submitted:
                st.error("❌ Por favor, digite um tema")
    
    @staticmethod
    def _generate_theme_flashcards(theme, num_cards, difficulty, category):
        """Gera flashcards baseados apenas no tema"""
        client = OpenAIService.get_client()
        
        if not client:
            # Gerar flashcards de exemplo sem API
            FlashcardsByTheme._generate_example_flashcards(theme, num_cards, category)
            return
        
        try:
            with st.spinner(f"🧠 Gerando {num_cards} flashcards sobre {theme}..."):
                prompt = f"""
                Crie {num_cards} flashcards educativos sobre "{theme}".
                Nível: {difficulty}
                
                IMPORTANTE: Retorne APENAS um array JSON no formato:
                [{{"q": "pergunta clara e específica", "a": "resposta completa e educativa"}}]
                
                Requisitos:
                - Perguntas progressivas do básico ao avançado
                - Respostas detalhadas e educativas
                - Cubra diferentes aspectos do tema
                - Use exemplos quando apropriado
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Você é um professor especializado criando flashcards educativos."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.7
                )
                
                # Processar resposta
                content = response.choices[0].message.content
                flashcards_data = FlashcardsByTheme._parse_flashcards_response(content)
                
                # Salvar flashcards
                saved_count = 0
                for card_data in flashcards_data[:num_cards]:
                    if 'q' in card_data and 'a' in card_data:
                        card_id = Utils.generate_id(card_data['q'])
                        
                        flashcard = Flashcard(
                            id=card_id,
                            question=card_data['q'],
                            answer=card_data['a'],
                            difficulty={"Iniciante": 2, "Intermediário": 3, "Avançado": 4}[difficulty],
                            category=category,
                            last_reviewed="",
                            next_review="",
                            correct_count=0,
                            total_reviews=0
                        )
                        
                        UserAwareSessionManager.save_flashcard(flashcard) #EnhancedSessionManager.save_flashcard(flashcard) #st.session_state.flashcards[card_id] = flashcard
                        saved_count += 1
                
                st.success(f"✅ {saved_count} flashcards criados sobre {theme}!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Erro ao gerar flashcards: {str(e)}")
            # Fallback para exemplos
            FlashcardsByTheme._generate_example_flashcards(theme, num_cards, category)
    
    @staticmethod
    def _parse_flashcards_response(content):
        """Parse seguro da resposta JSON"""
        import json
        import re
        
        try:
            # Tentar parse direto
            return json.loads(content)
        except:
            # Extrair JSON com regex
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
        
        return []
    
    @staticmethod
    def _generate_example_flashcards(theme, num_cards, category):
        """Gera flashcards de exemplo quando API não está disponível"""
        example_cards = {
            "default": [
                {"q": f"O que é {theme}?", "a": f"Definição básica de {theme} seria..."},
                {"q": f"Quais são os principais conceitos de {theme}?", "a": f"Os conceitos fundamentais incluem..."},
                {"q": f"Como aplicar {theme} na prática?", "a": f"Aplicações práticas incluem..."},
                {"q": f"Quais são os benefícios de estudar {theme}?", "a": f"Os principais benefícios são..."},
                {"q": f"Qual a importância de {theme} atualmente?", "a": f"A relevância atual se deve a..."}
            ]
        }
        
        cards_to_create = example_cards["default"][:num_cards]
        
        for i, card_data in enumerate(cards_to_create):
            card_id = Utils.generate_id(f"{theme}_{i}")
            
            flashcard = Flashcard(
                id=card_id,
                question=card_data['q'],
                answer=card_data['a'],
                difficulty=3,
                category=category,
                last_reviewed="",
                next_review="",
                correct_count=0,
                total_reviews=0
            )
            
            UserAwareSessionManager.save_flashcard(flashcard) #EnhancedSessionManager.save_flashcard(flashcard) #st.session_state.flashcards[card_id] = flashcard
        
        st.success(f"✅ {len(cards_to_create)} flashcards de exemplo criados!")
        st.balloons()
        st.rerun()
        
class FlashcardsPage:
    """Página do sistema de flashcards"""
    
    @staticmethod
    def render():
        st.markdown("### 🗃️ Sistema de Flashcards Inteligente")
        
        #tab1, tab2, tab3, tab4 = st.tabs(["➕ Criar", "🤖 IA", "📖 Estudar", "📊 Progresso"])
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["➕ Criar", "🎯 Por Tema", "🤖 IA", "📖 Estudar", "📊 Progresso"])
        with tab1:
            FlashcardsPage._render_create_flashcard()
        
        with tab2:
            FlashcardsByTheme.render_theme_flashcards()
            
        with tab3:
            FlashcardsPage._render_ai_generation()
            
        with tab4:
            FlashcardsPage._render_study_session()
        
        with tab5:
            FlashcardsPage._render_progress()
    
    @staticmethod
    def _render_create_flashcard():
        """Renderiza criação de flashcard"""
        st.subheader("Criar Flashcard")
        
        with st.form("new_flashcard_form", clear_on_submit=True):
            question = st.text_area("Pergunta*", placeholder="Digite a pergunta")
            answer = st.text_area("Resposta*", placeholder="Digite a resposta")
            category = st.selectbox("Categoria", ["IA", "Machine Learning", "Programação", "Matemática", "Outros"])
            difficulty = st.slider("Dificuldade", 1, 5, 3, help="1=Muito Fácil, 5=Muito Difícil")
            
            submitted = st.form_submit_button("💾 Criar Flashcard", use_container_width=True)
            
            if submitted:
                if question and answer:
                    card_id = Utils.generate_id(question)
                    
                    flashcard = Flashcard(
                        id=card_id,
                        question=question.strip(),
                        answer=answer.strip(),
                        difficulty=difficulty,
                        category=category,
                        last_reviewed="",
                        next_review="",
                        correct_count=0,
                        total_reviews=0
                    )
                    
                    UserAwareSessionManager.save_flashcard(flashcard) #EnhancedSessionManager.save_flashcard(flashcard) #st.session_state.flashcards[card_id] = flashcard
                    st.success("✅ Flashcard criado!")
                    st.rerun()
                else:
                    st.error("❌ Preencha pergunta e resposta")
    
    @staticmethod
    def _render_ai_generation():
        """Renderiza geração de flashcards com IA"""
        st.subheader("Gerar Flashcards com IA")
        
        if not st.session_state.notes:
            st.info("📝 Crie algumas notas primeiro para gerar flashcards automaticamente!")
            return
        
        selected_notes = st.multiselect(
            "Selecione notas para gerar flashcards",
            options=list(st.session_state.notes.keys()),
            format_func=lambda x: st.session_state.notes[x].title
        )
        
        num_cards = st.slider("Número de flashcards", 1, 10, 5)
        
        if st.button("🤖 Gerar com IA", use_container_width=True):
            if selected_notes:
                with st.spinner("🧠 Gerando flashcards com IA..."):
                    content = " ".join([st.session_state.notes[note_id].content for note_id in selected_notes])
                    generated_cards = OpenAIService.generate_flashcards(content, num_cards)
                    
                    for card_data in generated_cards:
                        card_id = Utils.generate_id(card_data['q'])
                        
                        flashcard = Flashcard(
                            id=card_id,
                            question=card_data['q'],
                            answer=card_data['a'],
                            difficulty=3,
                            category="IA",
                            last_reviewed="",
                            next_review="",
                            correct_count=0,
                            total_reviews=0
                        )
                        
                        UserAwareSessionManager.save_flashcard(flashcard) #EnhancedSessionManager.save_flashcard(flashcard) #st.session_state.flashcards[card_id] = flashcard
                    
                    st.success(f"✅ {len(generated_cards)} flashcards gerados!")
                    st.rerun()
            else:
                st.error("❌ Selecione pelo menos uma nota")
    
    @staticmethod
    def _render_study_session():
        """Renderiza sessão de estudo"""
        st.subheader("Sessão de Estudo")
        
        if not st.session_state.flashcards:
            st.info("🗃️ Crie alguns flashcards primeiro!")
            return
        
        # Verificar se está em modo de estudo
        if not st.session_state.study_mode:
            category_filter = st.selectbox(
                "Categoria", 
                ["Todas"] + list(set([card.category for card in st.session_state.flashcards.values()]))
            )
            
            if st.button("🚀 Iniciar Sessão de Estudo", use_container_width=True):
                available_cards = [
                    card for card in st.session_state.flashcards.values() 
                    if category_filter == "Todas" or card.category == category_filter
                ]
                
                if available_cards:
                    st.session_state.study_mode = True
                    st.session_state.current_card = random.choice(available_cards)
                    st.session_state.show_answer = False
                    st.rerun()
                else:
                    st.error("❌ Nenhum flashcard encontrado para esta categoria")
        else:
            # Modo de estudo ativo
            card = st.session_state.current_card
            
            # Mostrar flashcard
            if st.session_state.show_answer:
                st.markdown(f"""
                <div class='flashcard'>
                    <div>
                        <h4>❓ {card.question}</h4>
                        <hr style="border-color: white; margin: 1rem 0;">
                        <h4>✅ {card.answer}</h4>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='flashcard'>
                    <h4>❓ {card.question}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # Controles
            if not st.session_state.show_answer:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("👁️ Mostrar Resposta", use_container_width=True):
                        st.session_state.show_answer = True
                        st.rerun()
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("👎 Difícil", use_container_width=True):
                        card.total_reviews += 1
                        FlashcardsPage._next_card()
                
                with col2:
                    if st.button("➡️ Próximo", use_container_width=True):
                        card.total_reviews += 1
                        FlashcardsPage._next_card()
                
                with col3:
                    if st.button("👍 Fácil", use_container_width=True):
                        card.correct_count += 1
                        card.total_reviews += 1
                        FlashcardsPage._next_card()
            
            # Botão para finalizar
            if st.button("🛑 Finalizar Sessão"):
                # Registrar sessão de estudo
                session = StudySession(
                    date=datetime.now().strftime("%Y-%m-%d"),
                    activity="Flashcards",
                    duration=15,
                    category=card.category,
                    score=0.8
                )
                UserAwareSessionManager.save_study_session(session) #EnhancedSessionManager.save_study_session(session) #st.session_state.study_sessions.append(session)
                
                # Resetar modo de estudo
                st.session_state.study_mode = False
                st.session_state.current_card = None
                st.session_state.show_answer = False
                st.rerun()
    
    @staticmethod
    def _next_card():
        """Seleciona próximo flashcard"""
        st.session_state.current_card = random.choice(list(st.session_state.flashcards.values()))
        st.session_state.show_answer = False
        st.rerun()
    
    @staticmethod
    def _render_progress():
        """Renderiza progresso dos flashcards"""
        st.subheader("Progresso nos Flashcards")
        
        if not st.session_state.flashcards:
            st.info("📊 Nenhum flashcard criado ainda!")
            return
        
        # Preparar dados
        cards_data = []
        for card in st.session_state.flashcards.values():
            cards_data.append({
                'Pergunta': Utils.truncate_text(card.question, 50),
                'Categoria': card.category,
                'Dificuldade': card.difficulty,
                'Acertos': card.correct_count,
                'Total': card.total_reviews,
                'Precisão (%)': round(card.accuracy, 1)
            })
        
        if cards_data:
            df = pd.DataFrame(cards_data)
            st.dataframe(df, use_container_width=True)
            
            # Gráfico de progresso por categoria
            if len(df) > 0:
                category_stats = df.groupby('Categoria').agg({
                    'Precisão (%)': 'mean',
                    'Total': 'sum'
                }).reset_index()
                
                fig = px.bar(
                    category_stats, 
                    x='Categoria', 
                    y='Precisão (%)', 
                    title="Precisão Média por Categoria"
                )
                st.plotly_chart(fig, use_container_width=True)

# Continuar implementação...

class QuizPage:
    """Página do sistema de simulados"""
    
    @staticmethod
    def render():
        st.markdown("### 📊 Sistema de Simulados")
        
        tab1, tab2, tab3 = st.tabs(["➕ Criar", "🤖 IA", "📝 Realizar"])
        
        with tab1:
            QuizPage._render_create_quiz()
        
        with tab2:
            QuizPage._render_ai_generation()
        
        with tab3:
            QuizPage._render_take_quiz()
    
    @staticmethod
    def _render_create_quiz():
        """Renderiza criação de simulado"""
        st.subheader("Criar Simulado Manual")
        
        # Inicializar questões temporárias
        if 'temp_quiz_questions' not in st.session_state:
            st.session_state.temp_quiz_questions = []
        
        # Formulário do quiz
        with st.form("quiz_info_form"):
            title = st.text_input("Título do simulado*")
            category = st.selectbox("Categoria", ["IA", "Machine Learning", "Programação", "Matemática", "Outros"])
            st.form_submit_button("Configurar Quiz")
        
        # Adicionar questões
        st.markdown("#### Adicionar Questões")
        
        with st.form("question_form", clear_on_submit=True):
            question = st.text_area("Pergunta*")
            
            col1, col2 = st.columns(2)
            with col1:
                option1 = st.text_input("Opção 1*")
                option3 = st.text_input("Opção 3*")
            with col2:
                option2 = st.text_input("Opção 2*")
                option4 = st.text_input("Opção 4*")
            
            correct = st.selectbox("Resposta correta", [1, 2, 3, 4]) - 1
            explanation = st.text_area("Explicação")
            
            if st.form_submit_button("➕ Adicionar Questão"):
                if question and option1 and option2 and option3 and option4:
                    options = [option1, option2, option3, option4]
                    st.session_state.temp_quiz_questions.append({
                        "question": question,
                        "options": options,
                        "correct": correct,
                        "explanation": explanation
                    })
                    st.success("✅ Questão adicionada!")
                    st.rerun()
                else:
                    st.error("❌ Preencha todos os campos obrigatórios")
        
        # Mostrar questões adicionadas
        if st.session_state.temp_quiz_questions:
            st.write(f"**Questões adicionadas:** {len(st.session_state.temp_quiz_questions)}")
            
            for i, q in enumerate(st.session_state.temp_quiz_questions):
                with st.expander(f"Questão {i+1}: {Utils.truncate_text(q['question'], 50)}"):
                    st.write(f"**Pergunta:** {q['question']}")
                    for j, option in enumerate(q['options']):
                        marker = "✅" if j == q['correct'] else "▫️"
                        st.write(f"{marker} {j+1}. {option}")
                    if q['explanation']:
                        st.write(f"**Explicação:** {q['explanation']}")
                    
                    if st.button(f"🗑️ Remover", key=f"remove_q_{i}"):
                        st.session_state.temp_quiz_questions.pop(i)
                        EnhancedSessionManager.delete_quiz(quiz_id)
                        st.rerun()
            
            # Salvar quiz
            if st.button("💾 Salvar Simulado", use_container_width=True):
                if title and st.session_state.temp_quiz_questions:
                    quiz_id = Utils.generate_id(title)
                    
                    quiz = Quiz(
                        id=quiz_id,
                        title=title,
                        questions=st.session_state.temp_quiz_questions.copy(),
                        category=category,
                        created_at=datetime.now().isoformat()
                    )
                    
                    UserAwareSessionManager.save_quiz(quiz) #EnhancedSessionManager.save_quiz(quiz) #st.session_state.quizzes[quiz_id] = quiz
                    st.session_state.temp_quiz_questions = []
                    st.success("✅ Simulado salvo!")
                    st.rerun()
                else:
                    st.error("❌ Preencha o título e adicione pelo menos uma questão")
    
    @staticmethod
    def _render_ai_generation():
        """Renderiza geração de simulado com IA"""
        st.subheader("Gerar Simulado com IA")
        
        with st.form("ai_quiz_form"):
            topic = st.text_input("Tópico do simulado*", placeholder="Ex: Redes Neurais")
            num_questions = st.slider("Número de questões", 1, 20, 5)
            difficulty = st.selectbox("Nível de dificuldade", ["Iniciante", "Intermediário", "Avançado"])
            
            submitted = st.form_submit_button("🤖 Gerar com IA", use_container_width=True)
            
            if submitted and topic:
                with st.spinner("🧠 Gerando simulado com IA..."):
                    generated_quiz = OpenAIService.generate_quiz(topic, num_questions)
                    
                    quiz_id = Utils.generate_id(topic)
                    
                    quiz = Quiz(
                        id=quiz_id,
                        title=f"Simulado IA: {topic}",
                        questions=generated_quiz,
                        category="IA",
                        created_at=datetime.now().isoformat()
                    )
                    
                    UserAwareSessionManager.save_quiz(quiz) #EnhancedSessionManager.save_quiz(quiz) #st.session_state.quizzes[quiz_id] = quiz
                    st.success(f"✅ Simulado gerado com {len(generated_quiz)} questões!")
                    st.rerun()
            elif submitted:
                st.error("❌ Digite um tópico")
    
    @staticmethod
    def _render_take_quiz():
        """Renderiza realização de simulado com verificação de estado"""
        st.subheader("Realizar Simulado")
        
        if not st.session_state.quizzes:
            st.info("📊 Crie alguns simulados primeiro!")
            return
        
        # Verificar se active_quiz existe e é válido
        if 'active_quiz' not in st.session_state or st.session_state.active_quiz not in st.session_state.quizzes:
            # Resetar estado se inválido
            if 'active_quiz' in st.session_state:
                del st.session_state.active_quiz
            if 'quiz_submitted' in st.session_state:
                del st.session_state.quiz_submitted
            if 'quiz_answers' in st.session_state:
                del st.session_state.quiz_answers
        
        # Selecionar quiz
        if 'active_quiz' not in st.session_state:
            quiz_options = {quiz.id: quiz.title for quiz in st.session_state.quizzes.values()}
            selected_quiz_id = st.selectbox(
                "Escolha um simulado", 
                options=list(quiz_options.keys()), 
                format_func=lambda x: quiz_options[x]
            )
            
            if selected_quiz_id and selected_quiz_id in st.session_state.quizzes:
                quiz = st.session_state.quizzes[selected_quiz_id]
                st.write(f"**Categoria:** {quiz.category}")
                st.write(f"**Questões:** {len(quiz.questions)}")
                st.write(f"**Criado em:** {Utils.format_date(quiz.created_at)}")
                
                if st.button("🚀 Iniciar Simulado", use_container_width=True):
                    st.session_state.active_quiz = selected_quiz_id
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()
        
        # Realizar quiz
        elif not st.session_state.get('quiz_submitted', False):
            quiz = st.session_state.quizzes[st.session_state.active_quiz]
            
            st.markdown(f"### 📝 {quiz.title}")
            st.progress(0.0)  # Placeholder para progresso
            
            with st.form("quiz_form"):
                for i, q in enumerate(quiz.questions):
                    st.markdown(f"**Questão {i+1}:** {q['question']}")
                    
                    answer = st.radio(
                        f"Opções para questão {i+1}:",
                        options=range(len(q['options'])),
                        format_func=lambda x, opts=q['options']: f"{x+1}. {opts[x]}",
                        key=f"q_{i}",
                        index=None
                    )
                    
                    if answer is not None:
                        st.session_state.quiz_answers[i] = answer
                    
                    st.markdown("---")
                
                if st.form_submit_button("✅ Finalizar Simulado", use_container_width=True):
                    QuizPage._process_quiz_results(quiz)
        
        # Mostrar resultados
        else:
            QuizPage._show_quiz_results()
    
    @staticmethod
    def _process_quiz_results(quiz):
        """Processa resultados do quiz"""
        correct = 0
        total = len(quiz.questions)
        
        for i, q in enumerate(quiz.questions):
            if st.session_state.quiz_answers.get(i) == q['correct']:
                correct += 1
        
        score = (correct / total) * 100 if total > 0 else 0
        
        # Registrar sessão
        session = StudySession(
            date=datetime.now().strftime("%Y-%m-%d"),
            activity="Simulado",
            duration=total * 2,  # 2 minutos por questão
            category=quiz.category,
            score=score / 100
        )
        UserAwareSessionManager.save_study_session(session) #EnhancedSessionManager.save_study_session(session) #st.session_state.study_sessions.append(session)
        
        # Salvar resultados
        st.session_state.quiz_submitted = True
        st.session_state.quiz_score = score
        st.session_state.quiz_correct = correct
        st.session_state.quiz_total = total
        st.rerun()
    
    @staticmethod
    def _show_quiz_results():
        """Mostra resultados do quiz"""
        score = st.session_state.quiz_score
        correct = st.session_state.quiz_correct
        total = st.session_state.quiz_total
        
        # Card de resultado
        color = "green" if score >= 70 else "orange" if score >= 50 else "red"
        st.markdown(f"""
        <div class='card' style='background: linear-gradient(135deg, {color} 0%, #764ba2 100%);'>
            <h2>🎯 Resultado do Simulado</h2>
            <h3>Pontuação: {score:.1f}%</h3>
            <p>Acertos: {correct}/{total}</p>
            <p>{'🌟 Excelente!' if score >= 80 else '👍 Bom trabalho!' if score >= 60 else '📚 Continue estudando!'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gabarito detalhado
        quiz = st.session_state.quizzes[st.session_state.active_quiz]
        
        st.markdown("### 📋 Gabarito Detalhado")
        
        for i, q in enumerate(quiz.questions):
            user_answer = st.session_state.quiz_answers.get(i)
            is_correct = user_answer == q['correct']
            
            icon = "✅" if is_correct else "❌"
            
            with st.expander(f"{icon} Questão {i+1} - {q['question'][:50]}..."):
                st.markdown(f"**Pergunta:** {q['question']}")
                
                # Mostrar opções com destaque
                for j, option in enumerate(q['options']):
                    if j == q['correct']:
                        st.markdown(f"✅ **{j+1}. {option}** (Correta)")
                    elif j == user_answer:
                        st.markdown(f"❌ **{j+1}. {option}** (Sua resposta)")
                    else:
                        st.markdown(f"▫️ {j+1}. {option}")
                
                if q.get('explanation'):
                    st.markdown(f"**💡 Explicação:** {q['explanation']}")
        
        # Botões de ação
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Fazer Outro Simulado", use_container_width=True):
                del st.session_state.active_quiz
                del st.session_state.quiz_submitted
                st.rerun()
        
        with col2:
            if st.button("📊 Ver Progresso", use_container_width=True):
                st.session_state.current_page = "Progresso"
                st.rerun()

class UploadPage:
    """Página de upload e transcrição"""
    
    @staticmethod
    def render():
        st.markdown("### 📁 Upload e Transcrição de Materiais")
        
        tab1, tab2 = st.tabs(["📷 Transcrever Imagem", "📄 Upload de Arquivo"])
        
        with tab1:
            UploadPage._render_image_transcription()
        
        with tab2:
            UploadPage._render_file_upload()
    
    @staticmethod
    def _render_image_transcription():
        """Renderiza transcrição de imagem"""
        st.subheader("Transcrever Anotações de Imagem")
        st.info("📝 Faça upload de uma imagem com texto manuscrito ou digitado para extrair o conteúdo")
        
        uploaded_image = st.file_uploader(
            "Escolha uma imagem", 
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Formatos suportados: PNG, JPG, JPEG, BMP, TIFF"
        )
        
        if uploaded_image is not None:
            try:
                image = Image.open(uploaded_image)
                
                # Mostrar imagem
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(image, caption="Imagem carregada", use_container_width=True)
                
                with col2:
                    st.write("**Informações da imagem:**")
                    st.write(f"Formato: {image.format}")
                    st.write(f"Tamanho: {image.size}")
                    st.write(f"Modo: {image.mode}")
                
                # Extrair texto
                if st.button("🔍 Extrair Texto", use_container_width=True):
                    with st.spinner("Extraindo texto da imagem..."):
                        extracted_text = Utils.extract_text_from_image(image)
                        
                        st.markdown("### 📝 Texto Extraído:")
                        st.text_area(
                            "Texto extraído", 
                            value=extracted_text, 
                            height=200,
                            help="Você pode editar o texto antes de salvar"
                        )
                        
                        # Opções de salvamento
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("💾 Salvar como Nota"):
                                note_id = Utils.generate_id("OCR")
                                
                                note = Note(
                                    id=note_id,
                                    title=f"Nota OCR - {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                                    content=extracted_text,
                                    tags=["OCR", "transcrição", "imagem"],
                                    connections=[],
                                    created_at=datetime.now().isoformat(),
                                    category="Outros"
                                )
                                
                                UserAwareSessionManager.save_note(note) #EnhancedSessionManager.save_note(note) #st.session_state.notes[note_id] = note
                                st.success("✅ Nota criada com sucesso!")
                        
                        with col2:
                            if st.button("🤖 Gerar Flashcards"):
                                with st.spinner("Gerando flashcards..."):
                                    generated_cards = OpenAIService.generate_flashcards(extracted_text, 3)
                                    
                                    for card_data in generated_cards:
                                        card_id = Utils.generate_id(card_data['q'])
                                        
                                        flashcard = Flashcard(
                                            id=card_id,
                                            question=card_data['q'],
                                            answer=card_data['a'],
                                            difficulty=3,
                                            category="OCR",
                                            last_reviewed="",
                                            next_review="",
                                            correct_count=0,
                                            total_reviews=0
                                        )
                                        
                                        UserAwareSessionManager.save_flashcard(flashcard) #EnhancedSessionManager.save_flashcard(flashcard) #st.session_state.flashcards[card_id] = flashcard
                                    
                                    st.success(f"✅ {len(generated_cards)} flashcards gerados!")
            
            except Exception as e:
                st.error(f"❌ Erro ao processar imagem: {str(e)}")
    
    @staticmethod
    def _render_file_upload():
        """Renderiza upload de arquivo"""
        st.subheader("Upload de Materiais de Estudo")
        st.info("📄 Faça upload de arquivos PDF, Word, texto ou markdown")
        
        uploaded_file = st.file_uploader(
            "Escolha um arquivo", 
            type=['pdf', 'txt', 'docx', 'md', 'rtf'],
            help="Formatos suportados: PDF, TXT, DOCX, MD, RTF"
        )
        
        if uploaded_file is not None:
            # Informações do arquivo
            file_details = {
                "Nome": uploaded_file.name,
                "Tipo": uploaded_file.type,
                "Tamanho": f"{uploaded_file.size} bytes"
            }
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Detalhes do arquivo:**")
                for key, value in file_details.items():
                    st.write(f"**{key}:** {value}")
            
            with col2:
                if st.button("📚 Processar Arquivo", use_container_width=True):
                    with st.spinner("Processando arquivo..."):
                        # Simular processamento de arquivo
                        content = UploadPage._process_file(uploaded_file)
                        
                        st.markdown("### 📄 Conteúdo Processado:")
                        processed_content = st.text_area(
                            "Conteúdo extraído", 
                            value=content, 
                            height=300,
                            help="Você pode editar o conteúdo antes de salvar"
                        )
                        
                        # Opções de processamento
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("💾 Salvar como Nota"):
                                note_id = Utils.generate_id(uploaded_file.name)
                                
                                note = Note(
                                    id=note_id,
                                    title=f"Material: {uploaded_file.name}",
                                    content=processed_content,
                                    tags=["upload", "material", "arquivo"],
                                    connections=[],
                                    created_at=datetime.now().isoformat(),
                                    category="Outros"
                                )
                                
                                UserAwareSessionManager.save_note(note)# EnhancedSessionManager.save_note(note) #st.session_state.notes[note_id] = note
                                st.success("✅ Nota criada!")
                        
                        with col2:
                            if st.button("🤖 Gerar Flashcards"):
                                with st.spinner("Gerando flashcards..."):
                                    generated_cards = OpenAIService.generate_flashcards(processed_content, 5)
                                    
                                    for card_data in generated_cards:
                                        card_id = Utils.generate_id(card_data['q'])
                                        
                                        flashcard = Flashcard(
                                            id=card_id,
                                            question=card_data['q'],
                                            answer=card_data['a'],
                                            difficulty=3,
                                            category="Material Upload",
                                            last_reviewed="",
                                            next_review="",
                                            correct_count=0,
                                            total_reviews=0
                                        )
                                        
                                        UserAwareSessionManager.save_flashcard(flashcard)# EnhancedSessionManager.save_flashcard(flashcard) #st.session_state.flashcards[card_id] = flashcard
                                    
                                    st.success(f"✅ {len(generated_cards)} flashcards gerados!")
                        
                        with col3:
                            if st.button("📊 Gerar Quiz"):
                                with st.spinner("Gerando quiz..."):
                                    topic = uploaded_file.name.split('.')[0]
                                    generated_quiz = OpenAIService.generate_quiz(topic, 5)
                                    
                                    quiz_id = Utils.generate_id(topic)
                                    
                                    quiz = Quiz(
                                        id=quiz_id,
                                        title=f"Quiz: {topic}",
                                        questions=generated_quiz,
                                        category="Upload",
                                        created_at=datetime.now().isoformat()
                                    )
                                    
                                    UserAwareSessionManager.save_quiz(quiz) #EnhancedSessionManager.save_quiz(quiz) #st.session_state.quizzes[quiz_id] = quiz
                                    st.success("✅ Quiz gerado!")
    
    @staticmethod
    def _process_file(uploaded_file) -> str:
        """Processa arquivo uploadado e extrai conteúdo"""
        try:
            if uploaded_file.type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            
            elif uploaded_file.type == "text/markdown":
                return str(uploaded_file.read(), "utf-8")
            
            else:
                # Para outros tipos, retornar conteúdo simulado
                return f"""Conteúdo processado do arquivo: {uploaded_file.name}

Este é um exemplo de conteúdo extraído do arquivo carregado. 

Em uma implementação real, aqui seria extraído o texto do PDF, Word ou outros formatos usando bibliotecas específicas como:
- PyPDF2 ou pdfplumber para PDFs
- python-docx para arquivos Word
- Outras bibliotecas especializadas

O conteúdo real seria parseado e formatado adequadamente para estudo.

Tópicos identificados:
1. Conceito principal A
2. Conceito principal B  
3. Definições importantes
4. Exemplos práticos

Este texto pode ser usado para gerar notas, flashcards ou quizzes automaticamente."""
        
        except Exception as e:
            return f"Erro ao processar arquivo: {str(e)}"

# ===============================
# MÓDULO 4: NAVEGAÇÃO SIDEBAR
# ===============================

# def create_sidebar_navigation():
#     """Cria navegação na sidebar do Streamlit"""
#     with st.sidebar:
#         st.markdown("# 🧠 StudyAI")
#         st.markdown("---")
        
#         # Menu principal
#         st.markdown("### 📚 Menu Principal")
        
#         menu_items = [
#             ("🏠", "Dashboard", "Visão geral do seu progresso"),
#             ("📝", "Notas", "Criar e gerenciar notas"),
#             ("🗃️", "Flashcards", "Estudar com flashcards"),
#             ("📊", "Simulados", "Realizar quizzes"),
#             ("📁", "Upload", "Importar materiais"),
#             ("📈", "Progresso", "Análise detalhada"),
#             ("🗄️", "Banco de Dados", "Gerenciar dados")
#         ]
        
#         for icon, page, description in menu_items:
#             if st.button(
#                 f"{icon} {page}", 
#                 key=f"nav_{page}",
#                 use_container_width=True,
#                 help=description
#             ):
#                 st.session_state.current_page = page
#                 st.rerun()
        
#         # Indicador da página atual
#         current = st.session_state.get('current_page', 'Dashboard')
#         st.markdown(f"**Página atual:** {current}")
        
#         st.markdown("---")
        
#         # Estatísticas rápidas na sidebar
#         st.markdown("### 📊 Estatísticas Rápidas")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             notes_count = len(st.session_state.get('notes', {}))
#             st.metric("Notas", notes_count)
        
#         with col2:
#             cards_count = len(st.session_state.get('flashcards', {}))
#             st.metric("Cards", cards_count)
        
#         # Configurações
#         st.markdown("---")
#         st.markdown("### ⚙️ Configurações")
        
#         # API Key status
#         if st.session_state.get('openai_api_key'):
#             st.success("✅ OpenAI configurada")
#         else:
#             st.warning("⚠️ Configure OpenAI")
            
#         # Tema (exemplo)
#         theme = st.selectbox(
#             "Tema",
#             ["Claro", "Escuro", "Automático"],
#             key="theme_selector"
#         )

def create_authenticated_sidebar():
    """Cria sidebar para usuários autenticados"""
    with st.sidebar:
        # Cabeçalho com info do usuário
        user = st.session_state.current_user
        st.markdown(f"### 👤 Olá, {user.username}!")
        st.markdown("---")
        
        # Menu principal
        st.markdown("### 📚 Menu Principal")
        
        menu_items = [
            ("🏠", "Dashboard", "Visão geral do seu progresso"),
            ("📝", "Notas", "Criar e gerenciar notas"),
            ("🗃️", "Flashcards", "Estudar com flashcards"),
            ("📊", "Simulados", "Realizar quizzes"),
            ("📁", "Upload", "Importar materiais"),
            ("📈", "Progresso", "Análise detalhada"),
            ("👤", "Perfil", "Configurações da conta"),
        ]
        
        for icon, page, description in menu_items:
            if st.button(
                f"{icon} {page}", 
                key=f"nav_{page}",
                use_container_width=True,
                help=description
            ):
                st.session_state.current_page = page
                st.rerun()
        
        # Indicador da página atual
        current = st.session_state.get('current_page', 'Dashboard')
        st.markdown(f"**Página atual:** {current}")
        
        st.markdown("---")
        
        # Estatísticas rápidas
        st.markdown("### 📊 Estatísticas Rápidas")
        
        col1, col2 = st.columns(2)
        with col1:
            notes_count = len(st.session_state.get('notes', {}))
            st.metric("Notas", notes_count)
        
        with col2:
            cards_count = len(st.session_state.get('flashcards', {}))
            st.metric("Cards", cards_count)
        
        # Logout
        st.markdown("---")
        if st.button("🚪 Sair", use_container_width=True):
            # Limpar dados do usuário
            keys_to_remove = [
                'current_user', 'user_data_loaded', 'notes', 
                'flashcards', 'quizzes', 'study_sessions', 'openai_api_key'
            ]
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.session_state.logged_in = False
            st.rerun()
            
class ProgressPage:
    """Página de progresso e analytics"""
    
    @staticmethod
    def render():
        st.markdown("### 📈 Progresso e Analytics")
        
        if not st.session_state.study_sessions:
            st.info("📊 Nenhuma sessão de estudo registrada ainda. Comece a estudar para ver seu progresso!")
            ProgressPage._render_quick_start()
            return
        
        # Métricas gerais
        ProgressPage._render_general_metrics()
        
        # Gráficos de progresso
        ProgressPage._render_progress_charts()
        
        # Análise detalhada
        ProgressPage._render_detailed_analysis()
    
    @staticmethod
    def _render_quick_start():
        """Renderiza guia de início rápido"""
        st.markdown("### 🚀 Como começar:")
        
        steps = [
            ("📝", "Crie algumas notas", "Vá para a aba 'Notas' e adicione conteúdo"),
            ("🗃️", "Gere flashcards", "Use IA para criar flashcards das suas notas"),
            ("📖", "Estude com flashcards", "Faça sessões de estudo regulares"),
            ("📊", "Realize simulados", "Teste seu conhecimento com quizzes"),
            ("📈", "Acompanhe progresso", "Volte aqui para ver sua evolução")
        ]
        
        for icon, title, description in steps:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"<h2>{icon}</h2>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{title}**")
                st.write(description)
    
    @staticmethod
    def _render_general_metrics():
        """Renderiza métricas gerais"""
        tracker = AdvancedProgressTracker(
            st.session_state.study_sessions,
            st.session_state.flashcards,
            st.session_state.quizzes
        )
        metrics = tracker.calculate_comprehensive_metrics()
        figures = tracker.generate_advanced_visualizations()
    
    @staticmethod
    def _render_progress_charts():
        """Renderiza gráficos de progresso"""
        sessions = st.session_state.study_sessions
        df = pd.DataFrame([session for session in sessions])
        
        # Converter datas
        df['date'] = pd.to_datetime(df['date'])
        
        # Gráficos lado a lado
        col1, col2 = st.columns(2)
        
        with col1:
            # Evolução da pontuação
            fig1 = px.line(
                df, 
                x='date', 
                y='score', 
                color='activity',
                title="📈 Evolução da Pontuação",
                labels={'score': 'Pontuação', 'date': 'Data'}
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Tempo por categoria
            category_time = df.groupby('category')['duration'].sum().reset_index()
            fig2 = px.pie(
                category_time, 
                values='duration', 
                names='category',
                title="⏱️ Tempo por Categoria"
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Gráfico de atividade diária
        daily_activity = df.groupby([df['date'].dt.date, 'activity'])['duration'].sum().reset_index()
        daily_activity['date'] = pd.to_datetime(daily_activity['date'])
        
        fig3 = px.bar(
            daily_activity, 
            x='date', 
            y='duration', 
            color='activity',
            title="📅 Atividade Diária",
            labels={'duration': 'Duração (min)', 'date': 'Data'}
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    @staticmethod
    def _render_detailed_analysis():
        """Renderiza análise detalhada"""
        st.markdown("### 📊 Análise Detalhada")
        
        sessions = st.session_state.study_sessions
        df = pd.DataFrame([session for session in sessions])
        
        # Análise por atividade
        activity_stats = df.groupby('activity').agg({
            'duration': ['sum', 'mean', 'count'],
            'score': 'mean'
        }).round(2)
        
        activity_stats.columns = ['Tempo Total (min)', 'Tempo Médio (min)', 'Sessões', 'Pontuação Média']
        
        st.markdown("#### 📈 Estatísticas por Atividade")
        st.dataframe(activity_stats, use_container_width=True)
        
        # Tendências
        st.markdown("#### 📊 Tendências")
        
        # Calcular streaks
        dates = sorted(set([s.date for s in sessions]))
        current_streak = ProgressPage._calculate_streak(dates)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_category = df.groupby('category')['score'].mean().idxmax()
            best_score = df.groupby('category')['score'].mean().max()
            st.metric("🏆 Melhor Categoria", best_category, f"{best_score:.1%}")
        
        with col2:
            st.metric("🔥 Sequência Atual", f"{current_streak} dias")
        
        with col3:
            total_hours = df['duration'].sum() / 60
            st.metric("📚 Total de Horas", f"{total_hours:.1f}h")
        
        # Metas e conquistas
        ProgressPage._render_achievements()
    
    @staticmethod
    def _calculate_streak(dates):
        """Calcula sequência atual de dias estudando"""
        if not dates:
            return 0
        
        dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in dates]
        dates = sorted(set(dates), reverse=True)
        
        today = current_date.today()
        streak = 0
        
        for i, date in enumerate(dates):
            expected_date = today - timedelta(days=i)
            if date == expected_date:
                streak += 1
            else:
                break
        
        return streak
    
    @staticmethod
    def _render_achievements():
        """Renderiza conquistas"""
        st.markdown("#### 🏆 Conquistas")
        
        sessions = st.session_state.study_sessions
        notes_count = len(st.session_state.notes)
        flashcards_count = len(st.session_state.flashcards)
        quizzes_count = len(st.session_state.quizzes)
        
        achievements = [
            (notes_count >= 1, "📝", "Primeira Nota", "Criou sua primeira nota"),
            (notes_count >= 10, "📚", "Colecionador", "Criou 10 notas"),
            (flashcards_count >= 5, "🗃️", "Estudioso", "Criou 5 flashcards"),
            (len(sessions) >= 5, "⭐", "Dedicado", "Completou 5 sessões de estudo"),
            (any(s.score >= 0.9 for s in sessions), "🎯", "Precisão", "Pontuação de 90%+ em um simulado"),
            (len(sessions) >= 10, "🏆", "Mestre", "Completou 10 sessões de estudo"),
        ]
        
        earned = [a for a in achievements if a[0]]
        pending = [a for a in achievements if not a[0]]
        
        if earned:
            st.markdown("**✅ Conquistadas:**")
            for _, icon, title, desc in earned:
                st.success(f"{icon} **{title}** - {desc}")
        
        if pending:
            st.markdown("**🔒 Próximas conquistas:**")
            for _, icon, title, desc in pending[:3]:  # Mostrar apenas as próximas 3
                st.info(f"{icon} **{title}** - {desc}")

# ===============================
# APLICAÇÃO PRINCIPAL
# ===============================

def create_authenticated_sidebar():
    """Cria sidebar para usuários autenticados"""
    with st.sidebar:
        # Cabeçalho com info do usuário
        user = st.session_state.current_user
        st.markdown(f"### 👤 Olá, {user.username}!")
        st.markdown("---")
        
        # Menu principal
        st.markdown("### 📚 Menu Principal")
        
        menu_items = [
            ("🏠", "Dashboard", "Visão geral do seu progresso"),
            ("📝", "Notas", "Criar e gerenciar notas"),
            ("🗃️", "Flashcards", "Estudar com flashcards"),
            ("📊", "Simulados", "Realizar quizzes"),
            ("📁", "Upload", "Importar materiais"),
            ("📈", "Progresso", "Análise detalhada"),
            ("👤", "Perfil", "Configurações da conta"),
        ]
        
        for icon, page, description in menu_items:
            if st.button(
                f"{icon} {page}", 
                key=f"nav_{page}",
                use_container_width=True,
                help=description
            ):
                st.session_state.current_page = page
                st.rerun()
        
        # Indicador da página atual
        current = st.session_state.get('current_page', 'Dashboard')
        st.markdown(f"**Página atual:** {current}")
        
        st.markdown("---")
        
        # Estatísticas rápidas
        st.markdown("### 📊 Estatísticas Rápidas")
        
        col1, col2 = st.columns(2)
        with col1:
            notes_count = len(st.session_state.get('notes', {}))
            st.metric("Notas", notes_count)
        
        with col2:
            cards_count = len(st.session_state.get('flashcards', {}))
            st.metric("Cards", cards_count)
        
        # Logout
        st.markdown("---")
        if st.button("🚪 Sair", use_container_width=True):
            # Limpar dados do usuário
            keys_to_remove = [
                'current_user', 'user_data_loaded', 'notes', 
                'flashcards', 'quizzes', 'study_sessions', 'openai_api_key'
            ]
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.session_state.logged_in = False
            st.rerun()


def main_authenticated():
    """Função principal para usuários autenticados"""
    # Carregar CSS
    load_css()
    
    # Criar sidebar
    create_authenticated_sidebar()
    
    # Roteamento de páginas
    page = st.session_state.current_page
    
    try:
        if page == "Dashboard":
            Dashboard.render()
        elif page == "Notas":
            NotesPage.render()
        elif page == "Flashcards":
            FlashcardsPage.render()
        elif page == "Simulados":
            QuizPage.render()
        elif page == "Upload":
            UploadPage.render()
        elif page == "Progresso":
            ProgressPage.render()
        elif page == "Perfil":
            UserProfilePage.render()
        else:
            st.error(f"Página '{page}' não encontrada")
            
    except Exception as e:
        st.error(f"Erro na página {page}: {str(e)}")
        st.exception(e)


def main():
    """Função principal da aplicação com autenticação"""
    # Configurar página
    st.set_page_config(
        page_title="StudyAI - Sistema de Estudos Inteligente",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    apply_all_fixes()
    # Inicializar session manager com suporte a usuários
    UserAwareSessionManager.init_session_state()
    
    # Verificar se está logado
    if not st.session_state.logged_in:
        # Mostrar páginas de autenticação
        load_css()
        
        st.markdown("<h1 class='main-header'>🧠 StudyAI - Sistema de Estudos Inteligente</h1>", unsafe_allow_html=True)
        
        # Renderizar página de auth apropriada
        auth_page = st.session_state.get('auth_page', 'login')
        
        if auth_page == 'login':
            AuthPages.render_login_page()
        elif auth_page == 'register':
            AuthPages.render_register_page()
        elif auth_page == 'forgot_password':
            AuthPages.render_forgot_password_page()
    else:
        # Usuário autenticado - mostrar app principal
        main_authenticated()
        
if __name__ == "__main__":
    main()